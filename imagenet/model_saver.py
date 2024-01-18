import torchvision.models as models
import torch
import fire
import os

def save_model(device: str = "CPU", model_name = "resnet50"):
    """Serialize and save a Torchvision model

    Torchvision comes with a number of imagenet models each with a different
    architecture and number of layers. Common ones (and the number of parameters)
    are "shufflenet_v2_x0_5" (1.4M), "resnet50" (25.6M), and "resnet152" (60.2M).

    :param device: device to target for inference "CPU" or "GPU", defaults to "CPU"
    :type device: str, optional
    :param model_name: the Torchvision model to use, defaults to "resnet50"
    :type model_name: str, optional
    """
    print(f"Saving model for {device}")
    if device.startswith("GPU"):
        if not torch.cuda.is_available():
            raise Exception("Requested to create GPU model, but CUDA was not found.")
    # Device's torch name
    device_name = "cuda" if device.startswith("GPU") else "cpu"

    model = getattr(models, model_name)(pretrained=True)
    model.to(torch.device(device_name))
    model.eval()

    batch = torch.randn((1, 3, 224, 224), device=device_name)
    traced_model = torch.jit.trace(model, batch)

    path = os.path.dirname(os.path.realpath(__file__))
    torch.jit.save(traced_model, os.path.join(path, f'resnet50.{device[0:3]}.pt'))
    print("model saved")
    del model

if __name__ == '__main__':
    print("Welcome to model saver!")
    fire.Fire(save_model)

