from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from collections import namedtuple

from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.grad_sample import GradSampleModule
#https://opacus.ai/tutorials/guide_to_grad_sampler

from neural_nets_training.utils import get_new_optimizer

privacy_data = namedtuple("privacy_data",
                          ["privacy_engine", "epsilon", "delta"])


def privatize(*, model, data_loader, epochs, epsilon, delta, max_grad_norm,
              device):
    """ take a model, and returns a epsilon delta version of that NN model, implementing DP-SGD 
    return model, optimizer, data_loader
    """
    model = ModuleValidator.fix(model)
    model = model.to(device)
    ModuleValidator.validate(model, strict=True)
    optimizer = get_new_optimizer(
        model
    )  # note : if in DP mode, need to fix the model, before can define the optimizer (both before hte privacy engine stuff)

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")
    ModuleValidator.validate(model, strict=True)
    return privacy_engine, model, optimizer, data_loader