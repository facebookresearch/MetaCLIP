# Copyright (c) Meta Platforms, Inc. and affiliates

def detect_unused_parameters(model):
    found = False
    for name, param in model.named_parameters():
        if param.grad is None and param.requires_grad is True:
            found = True
            print(f"unused parameter: {name}")
    if found:
        exit(1)


def detect_nan(model, optimizer):
    try:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    # check local gradnorm single GPU case, trigger NanDetector
                    raise FloatingPointError(f"gradients are Nan/Inf on {name}")
    except FloatingPointError as e:
        logging.warn(f"{str(e)}, skip batch.")
        optimizer.zero_grad()

