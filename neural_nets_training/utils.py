# from pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

import torch
import yaml
import logging
from pathlib import Path
# privacy

# https://opacus.ai/tutorials/building_text_classifier


def profile_function(f, max_function_calls=100):
    """ 
    decorator to profile a function 
    """
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    f()

    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(max_function_calls)
    ret = s.getvalue()
    print(ret)
    return ret


def open_yaml(filename):
    """ return the contents of yaml as a dictionary """
    path = Path(filename)
    if not path.is_file():
        return None
    try:
        with open(path, "r") as infile:
            return yaml.safe_load(infile)
    except Exception as e:
        logging.info(e)
    return {}


def write_yaml(filename, d):
    """ dump contents of d into filename, in yaml format"""
    with open(filename, 'w') as file:
        yaml.dump(d, file, default_flow_style=False)


def write_pickle(filename, data):
    """ dump contents of d into filename, in pickle format """
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def compute_histogram(dl):
    count = 0
    hist = defaultdict(lambda: 0)
    for images, labels in dl:
        for label in labels:
            count += 1
            hist[label.item()] += 1
        if count % 10000 == 0:
            print(count)
    return hist


def plot_histogram(dataloader):
    # show histogram
    train_hist = compute_histogram(dataloader)
    lists = sorted(
        train_hist.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    y_sorted = sorted(y, reverse=True)
    plt.plot(np.arange(len(y)), y_sorted)
    plt.title("histogram of data labels")
    plt.show()


def grow_and_shrink_lr(optimizer, max_lr, total_epochs):
    """
    grows to max_lr, linearly in the first 15% of epochs,
    then shrinks linear to 0 in the remaining (as done in feldman)
        http://vtaly.net/papers/FZ_Infl_mem.pdf
    """
    peak_epoch = int(total_epochs * .15)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=1e-4,
                                                  end_factor=max_lr,
                                                  total_iters=peak_epoch)

    scheduler_2 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                    start_factor=max_lr,
                                                    end_factor=max_lr,
                                                    total_iters=total_epochs -
                                                    peak_epoch)
    schedulers = [scheduler, scheduler_2]
    return torch.optim.lr_scheduler.ChainedScheduler(schedulers)


def get_new_optimizer(model, max_lr=1e-2, weight_decay=1e-4):
    optimizer_type = torch.optim.Adam

    return optimizer_type(model.parameters(),
                          lr=max_lr,
                          weight_decay=weight_decay)


def save_np_to_filename(*, filepath, arr, overwrite=False):
    if os.path.exists(filepath) and not overwrite:
        print("file already exists, ignoring")
        return
    np.save(filepath, arr)


def load_np_from_filename(filepath):
    if not os.path.exists(filepath):
        print("file does not exist")
        return None
    return np.load(filepath)


def save_model(model, filepath):
    """ helper function to save model"""
    #torch.save(model.state_dict(), 'cifar100-resnet12layers.pth')
    torch.save(model.state_dict(), filepath)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def load_model(path, base_model, device, map_location=None):
    """ helper function to load model"""
    model = base_model
    loaded_dict = torch.load(path, map_location=map_location)
    """
    for name, buffer in model.named_buffers():
      if name not in loaded_dict:
        loaded_dict[name] = buffer
    """
    def rename(k):
        pref = "_module."
        if k[:len(pref)] == pref:
            return k[len(pref):]
        return k

    ld = {rename(k): v for k, v in loaded_dict.items()}

    model.load_state_dict(ld, strict=False)
    model = to_device(model, device)
    return model
