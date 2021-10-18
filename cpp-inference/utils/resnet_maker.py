import torch
import torchvision

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Creating model for {device}")

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).to(device=device)
        self.resnet = torch.jit.trace(torchvision.models.resnet50().to(device=device),
                                      torch.rand(1, 3, 224, 224).to(device=device))

    def forward(self, input):
        return self.resnet(input - self.means)

my_script_module = torch.jit.script(MyScriptModule())
torch.jit.save(my_script_module, "resnet_model.pt")