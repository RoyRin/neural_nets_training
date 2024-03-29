# from pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from opacus.utils.batch_memory_manager import BatchMemoryManager

from neural_nets_training import privacy
from neural_nets_training import utils
from neural_nets_training import params

MAX_PHYSICAL_BATCH_SIZE = 128
NO_DP_FLAG = -9999


def epoch_end(*, result):
    """Helper, print epoch results
    """
    epoch = result.get("epoch")
    accuracies = {k: v for k, v in result.items() if "accuracy" in k}
    losses = {k: v for k, v in result.items() if "loss" in k}
    print(f"Epoch [{epoch}], ")
    print(
        f"\tAccuracy: { ', '.join(f'{k}: {v:.4f}' for k, v in sorted(accuracies.items())) }"
    )
    print(
        f"\tLoss: { ', '.join(f'{k}: {v:.4f}' for k, v in sorted(losses.items())) }"
    )


def get_num_correct(outputs, targets):
    """ returns the number of predictions that are equal to the labels"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item()).item()


def compute_accuracy(outputs, targets):
    """return accuracy of the model, given targets """
    return get_num_correct(outputs, targets) / len(targets)


def prediction_by_imgs(*, model, imgs, device=params.get_default_device()):
    """
    returns confidence, prediction
    """
    imgs = utils.to_device(imgs, device)
    # if using individual img, use `img.unsqueeze_(0)`
    with torch.no_grad():
        yb = model(imgs)
    softmax = torch.nn.functional.softmax(
        yb, dim=1)  # for individual img, use yb[0]
    confs, preds = torch.max(softmax, dim=1)
    return confs, preds


def prediction_from_logits(logits):
    """ Similar to prediction by imgs, but with logits """
    softmax = torch.nn.functional.softmax(logits, dim=1)
    confs, preds = torch.max(softmax, dim=1)
    return confs, preds


@torch.no_grad()
def compute_probability_of_correct_labels(*, model, data_loader, device):
    """ Computes the probability the model correctly labels a point on the training set
    
    Args:
        model ([type]): [description]
        data_loader ([type]): [description]
        device ([type]): [description]

    Returns:
        confidences [np.array], accuracy
            accuracy [float from 0 to 1]
            confidences [np.array of floats from 0 to 1]
    """
    total = correct = 0
    N = len(data_loader.dataset)
    confidences = np.zeros(N)

    for ind, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        output = model(data)  # make prediction
        softmax = torch.nn.functional.softmax(output, dim=0)
        w = len(targets)
        confs = softmax[np.arange(len(softmax)), targets]
        confidences[w * ind:w * (ind + 1)] = utils.to_device(confs, "cpu")

        _, pred = torch.max(output, 1)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    return confidences, correct / total


def validation_step(*, model, batch, criterion, device, class_count=10):
    """validation step - evaluate model on a batch

    Args:
        model (_type_): _description_
        batch (_type_): _description_
        criterion (_type_): _description_
        device (_type_): _description_

    Returns:
        dict of stats: {loss, num_correct, num}
    """
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, labels).detach().item()  # Calculate loss
    num_correct = get_num_correct(out, labels)
    _, predictions = prediction_from_logits(logits=out.cpu())
    histogram = np.histogram(predictions, np.arange(class_count + 1))[0]

    return {
        'loss': loss,
        'num_correct': num_correct,
        "num": len(labels),
        "histogram": histogram
    }


def compute_epoch_accuracy(outputs):
    """
    returns accuracy of the model on the training set
    args:
        outputs: [list of dicts{}], computed by the output of validation_step
    """

    batch_losses = [x['loss'] for x in outputs]
    epoch_loss = sum(batch_losses) / len(batch_losses)  # Combine losses

    num_correct = sum([x['num_correct'] for x in outputs])
    total_count = sum([x['num'] for x in outputs])

    histograms = sum([x.get("histogram") for x in outputs])

    return {
        'loss': epoch_loss,
        'accuracy': num_correct / total_count,
        "histograms": histograms
    }


@torch.no_grad()
def evaluate(*, model, data_loader, criterion, device, class_count=10):
    """evaluate a model on a dataloader

    Args:
        model (_type_): _description_
        data_loader (_type_): _description_
        criterion (_type_): _description_
        device (_type_): _description_

    Returns:
        list of dicts: 
    """
    model.eval()
    outputs = [
        validation_step(model=model,
                        batch=batch,
                        criterion=criterion,
                        device=device,
                        class_count=class_count) for batch in data_loader
    ]
    return compute_epoch_accuracy(outputs)


@torch.no_grad()
def get_predictions(*, model, data_loader, device, optimizer=None):
    """ Computes the probability the model correctly labels a point on the training set
    
    Args:
        model ([type]): [description]
        data_loader ([type]): [description]
        device ([type]): [description]

    Returns:
        predictions [np.array], accuracy [float] 
            accuracy [float from 0 to 1]
    """
    if optimizer is not None:
        optimizer.zero_grad()
    total = correct = 0
    N = len(data_loader.dataset)
    predictions = np.zeros(N, dtype=np.int)
    current_ind = 0
    for ind, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)  # make prediction
        w = len(targets)
        _, preds = torch.max(outputs, dim=1)

        predictions[current_ind:current_ind + w] = preds.cpu().numpy()
        current_ind += w
        total += targets.size(0)
        correct += (preds == targets).sum().item()
    return predictions, correct / total


def select_n_random(dataset, n=100):
    ''' Selects n random datapoints and their corresponding labels from a dataset '''
    data = dataset.data
    perm = torch.randperm(len(data))
    rand_labels = [dataset.targets[i] for i in perm][:n]
    return data[perm][:n], rand_labels


def train_single_epoch(*,
                       model,
                       train_loader,
                       optimizer,
                       criterion,
                       device,
                       batch_size=params.batch_size):
    """ train a model for a single epoch 

    Args:
        model ([type]): [description]
        train_loader ([type]): [description]
        train_losses ([type]): [description]
        optimizer ([type]): [description]
        criterion ([type]): [description]
        device ([type]): [description]

    Returns:
        estimated_total_correct, estimated_train_losses: 
            total_correct: number of correct predictions
            train_losses: list of losses for each batch
    """
    model.train()

    estimated_total_correct = 0
    estimated_epoch_train_losses = []

    for _, (images, targets) in enumerate(train_loader):
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        optimizer.zero_grad()

        # do the training
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        # get stats
        correct = get_num_correct(output, targets)
        estimated_total_correct += correct
        # weight by the batch size
        weight = len(targets) / batch_size
        weighted_loss = loss.item() * weight
        estimated_epoch_train_losses.append(weighted_loss)

    return estimated_total_correct, estimated_epoch_train_losses


# Lookie here # https://jovian.ai/roubish/final-course-assignment
def train_model(
        *,
        model,
        model_name="",
        epochs,
        train_loader,
        device,
        test_loader=None,  # can be None - only called in per_epoch_callback hook
        max_grad_norm=params.MAX_GRAD_NORM,
        tensorboard_path=None,
        criterion,
        max_lr=.5,
        adam=True,  # Adam optimizer, if false, use SGD
        schedule_lr=True,
        eps=None,
        delta=None,
        per_epoch_callbacks=None,
        save_path=None,
        overwrite=False,
        verbose=False):
    """ train model for a number of epochs
        potentially allows for DP training
        will call each callback from per_epoch_callbacks at the end of each epoch
    Returns:
        ret [dict]: dictionary of results
                ret = {
                    "save_path": save_path,
                    "epsilon": eps,
                    "delta": delta,
                    "history": [
                        { dictionary with meta about each epoch }
                    ],
                    "model_name": model_name
                }
    """

    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
    # set up
    first_start = datetime.datetime.now()

    history = []

    ret = {
        "save_path": save_path,
        "privacy": {
            "epsilon": eps,
            "delta": delta
        },
        "history": history,
        "model_name": model_name
    }
    print(f"epsilon - {eps}")
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        print(f"path {save_path} already saved to. SKIPPING.")
        return ret
    writer = SummaryWriter(
        tensorboard_path) if tensorboard_path is not None else None

    per_epoch_callbacks = per_epoch_callbacks or []
    # set up privacy experiment if needed (use memory safe data loader as needed)
    optimizer = None
    if eps is not None:
        privacy_engine, model, optimizer, train_loader = privacy.privatize(
            model=model,
            data_loader=train_loader,
            epochs=epochs,
            epsilon=eps,
            delta=delta,
            max_grad_norm=max_grad_norm,
            device=device)

        def privacy_log_cb(*, result, verbose, **kwargs):
            """ log used epsilon"""
            if verbose:
                print("privacy used ", privacy_engine.get_epsilon(delta))
            result.update({"used_epsilon": privacy_engine.get_epsilon(delta)})

        per_epoch_callbacks.append(privacy_log_cb)

    model = model.to(device)
    optimizer = optimizer or utils.get_new_optimizer(
        adam=adam,  # if you schedule, you don't want to use adam
        momentum=0.9,
        weight_decay=0.9,
        model=model,
        max_lr=max_lr)  # take existing optimizer if provided
    learning_rate_scheduler = None

    if schedule_lr:
        learning_rate_scheduler = utils.grow_and_shrink_lr(optimizer=optimizer,
                                                           max_lr=max_lr,
                                                           total_epochs=epochs)

    # create writer

    def _train(dataloader):
        """ Helper to train either with DP or not (creates closure)"""
        for epoch in range(epochs):
            start = datetime.datetime.now()
            print(f"Training epoch: {epoch}/{epochs}")
            estimated_total_correct, estimated_epoch_train_losses = train_single_epoch(
                model=model,
                train_loader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device)

            # step through the learning rate scheduler if it exists
            if learning_rate_scheduler is not None:
                learning_rate_scheduler.step()

            result = {
                "epoch":
                epoch,
                "approx_train_loss":
                np.mean(estimated_epoch_train_losses),
                "approx_train_accuracy":
                estimated_total_correct / len(dataloader.dataset)
            }  # todo : do something more with result

            # call all callbacks
            for callback in per_epoch_callbacks:  # to do : check if callback is called on the last batch
                callback(model=model,
                         epoch=epoch,
                         train_loader=dataloader,
                         test_loader=test_loader,
                         optimizer=optimizer,
                         criterion=criterion,
                         result=result,
                         writer=writer,
                         device=device,
                         verbose=verbose)
            history.append(result)
            if verbose:
                epoch_end(result=result)
                print(
                    f"time for epoch {epoch} : {(datetime.datetime.now() - start)}"
                )
        # print the final epoch
        epoch_end(result=history[-1])
        return history

    # train with DP
    if eps is None:
        # train without DP
        ret["history"] = _train(train_loader)
    else:
        with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer) as new_data_loader:
            print(f"new data loader created {type(new_data_loader)}")
            ret["history"] = _train(new_data_loader)

    if save_path is not None:
        utils.save_model(model, save_path)

    print(f"total time: {(datetime.datetime.now() - first_start)}")
    if writer is not None:
        writer.flush()
    return ret


## Callbacks used in training
# All callbacks follow this pattern:
#     callback(model=model,
#                      epoch=epoch,
#                      train_loader=train_loader,
#                      test_loader=val_loader,
#                      optimizer=optimizer,
#                      criterion=criterion,
#                      result=result,
#                      writer=writer,
#                      device=device)
# 



def eval_validation_callback(*, result, model, test_loader, criterion, device,
                             **kwargs):
    """ Evaluate model on validation set 
        update result inline
    """
    class_count = len(test_loader.dataset.classes)
    val_result = evaluate(model=model,
                          data_loader=test_loader,
                          criterion=criterion,
                          device=device,
                          class_count=class_count)

    result.update({"val_" + k: v for k, v in val_result.items()})


def compute_probability_of_correct_labels_callback(*, model, train_loader,
                                                   result, device, **kwargs):
    """ Computes the probability the model correctly labels a point on the training set"""
    confidences, acc = compute_probability_of_correct_labels(
        model=model, data_loader=train_loader, device=device)
    result.update({"train_confidences": confidences})


def eval_training_callback(*, result, model, train_loader, criterion, device,
                           **kwargs):
    """ Evaluate model on training set 
        update result inline
    """
    class_count = len(train_loader.dataset.classes)
    train_result = evaluate(model=model,
                            data_loader=train_loader,
                            criterion=criterion,
                            device=device,
                            class_count=class_count)
    result.update({"train_" + k: v for k, v in train_result.items()})


def log_epoch_to_tensorboard(*, model, epoch, result, writer, **kwargs):
    """ log epoch data using the tensorboard writer """

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    for k, v in result.items():
        writer.add_scalar(k, v, epoch)
    return result


def log_model_to_tensorboard(*, model, dataset, writer, n=32, **kwargs):
    """ log model information to tensorboard (things like model graph and features) """
    # log to tensorboard writer
    # select random images and their target indices
    images, labels = select_n_random(dataset=dataset, n=n)
    writer.add_graph(model, images)


log_everything_callbacks = [
    eval_training_callback,
    eval_validation_callback,
    log_epoch_to_tensorboard,
    #log_model_to_tensorboard
]
""" needs to be of the form:
    #  callback(model=model,
                     epoch=epoch,
                     train_loader=train_loader,
                     test_loader=val_loader,
                     optimizer=optimizer,
                     criterion=criterion,
                     result=result,
                     writer=writer,
                     device=device)
"""
