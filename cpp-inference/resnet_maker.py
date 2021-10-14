import torch
import torchvision

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).cuda()
        self.resnet = torch.jit.trace(torchvision.models.resnet50().cuda(),
                                      torch.rand(1, 3, 224, 224).cuda())

    def forward(self, input):
        return self.resnet(input - self.means)

my_script_module = torch.jit.script(MyScriptModule())
torch.jit.save(my_script_module, "resnet_model.pt")