import torch
from torchinfo import summary
import json
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from pprint import pprint


def print_gpu_utilization(device_idx=0):
    print(f"GPUs used:\t{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    print(f"Device:\t\t{device}")

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def test_net(model, device="cpu", size=(3, 224, 224), n_batch=32, use_lt=False):
    model = model.to(device)
    model.eval()
    x = torch.randn(n_batch, *size, device=device)
    with torch.no_grad():
        output = model(x)
    if use_lt:
        print(
            "=========================================================================================="
        )
        print(f"Input shape: {x}")
        print(f"Output shape: {output}")
    else:
        print(
            "=========================================================================================="
        )
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    summary(model, input_size=(n_batch, *(size)), device=device)


def save_run_config(fname, config):
    consts = {}
    for k in dir(config):
        if k.isupper() and not k.startswith("_"):
            consts[k] = str(getattr(config, k))
    with open(f"{fname}.conf", "w") as f:
        f.write(json.dumps(obj=consts, indent=4))


def inspect_model(model, output="params"):
    """
    output: 'params' or 'state'
    """
    if output == "state":
        pprint(model.state_dict)
    elif output == "params":
        for idx, (name, param) in enumerate(model.named_parameters()):
            print(f"{idx}: {name} \n{param}")
            print(
                "------------------------------------------------------------------------------------------"
            )
    else:
        raise ValueError("Output must be either 'params' or 'state'")
