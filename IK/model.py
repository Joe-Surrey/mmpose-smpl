from torch import nn
import torch
import torchvision.models as models

class Model(nn.Module):
    """

    """
    def __init__(self, output_features=0, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.Model = models.resnet34(pretrained=True).to(self.device)
        self.Model.fc = nn.Linear(in_features=512, out_features=output_features).to(self.device)
        for parameter in self.Model.parameters():
            parameter.requires_grad = True
        #for parameter in self.Model.parameters():
        #    parameter.requires_grad = True
        #for parameter in self.Model.fc.parameters():
        #    parameter.requires_grad = True

    def forward(self, input):
        """
        Take in image


        """
        features = self.Model(input.to(self.device))

        return features.to('cpu')