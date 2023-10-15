
import torch
import torch as th
from torch import nn
from overridable_layers import OverLinear, OverConv2d, OverInstanceNorm2d


def original_oml_network(dataset_name, fixed_rln=True):
    if dataset_name == "omniglot":
        channels = 256
        representation_size=2304
        hidden_dim = 1024
        classes = 1000

        if fixed_rln:
            RLNInstanceNorm2d = nn.InstanceNorm2d
            RLNConv2d = nn.Conv2d
        else:
            RLNInstanceNorm2d = OverInstanceNorm2d
            RLNConv2d = OverConv2d

        rln_layers = [
            RLNConv2d(in_channels=3, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        ]
        pln_layers = [
                      OverLinear(in_features=representation_size,
                                out_features=hidden_dim),
                      nn.ReLU(inplace=True),
                      OverLinear(in_features=hidden_dim,
                                out_features=classes),
                      ]
        return nn.ModuleList(rln_layers), nn.ModuleList(pln_layers)


def original_anml_network(dataset_name, fixed_rln=False):
    if dataset_name == "omniglot":
        classes = 1000
        nm_channels = 112
        channels = 256
        size_of_representation = 2304
        size_of_interpreter = 1008

        if fixed_rln:
            RLNInstanceNorm2d = nn.InstanceNorm2d
            RLNConv2d = nn.Conv2d
        else:
            RLNInstanceNorm2d = OverInstanceNorm2d
            RLNConv2d = OverConv2d

        neuromod_layers = [
            nn.Conv2d(in_channels=3, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nm_channels, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nm_channels, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=size_of_interpreter,
                      out_features=size_of_representation),
            nn.Sigmoid(),
        ]

        rln_layers = [
            RLNConv2d(in_channels=3, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        ]

        pln_layers = [
            OverLinear(in_features=size_of_representation,
                      out_features=classes),
        ]
        return (nn.ModuleList(rln_layers), nn.ModuleList(pln_layers),
                nn.ModuleList(neuromod_layers))

def doublelinear_anml_network(dataset_name, fixed_rln=False):
    if dataset_name == "omniglot":
        classes = 1000
        nm_channels = 112
        channels = 256
        hidden_dim = 1024
        size_of_representation = 2304
        size_of_interpreter = 1008

        if fixed_rln:
            RLNInstanceNorm2d = nn.InstanceNorm2d
            RLNConv2d = nn.Conv2d
        else:
            RLNInstanceNorm2d = OverInstanceNorm2d
            RLNConv2d = OverConv2d

        neuromod_layers = [
            nn.Conv2d(in_channels=3, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nm_channels, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nm_channels, out_channels=nm_channels, kernel_size=3,
                      stride=1, padding=0),
            nn.InstanceNorm2d(num_features=nm_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=size_of_interpreter,
                      out_features=size_of_representation),
            nn.Sigmoid(),
        ]

        rln_layers = [
            RLNConv2d(in_channels=3, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        ]

        pln_layers = [
            OverLinear(in_features=size_of_representation,
                      out_features=hidden_dim),
            nn.ReLU(inplace=True),
            OverLinear(in_features=hidden_dim,
                       out_features=classes),
        ]
        return (nn.ModuleList(rln_layers), nn.ModuleList(pln_layers),
                nn.ModuleList(neuromod_layers))

def normalized_oml_network(dataset_name, fixed_rln=True):
    if dataset_name == "omniglot":
        channels = 256
        representation_size = 2304
        hidden_dim = 1024
        classes = 1000

        if fixed_rln:
            RLNInstanceNorm2d = nn.InstanceNorm2d
            RLNConv2d = nn.Conv2d
        else:
            RLNInstanceNorm2d = OverInstanceNorm2d
            RLNConv2d = OverConv2d

        rln_layers = [
            RLNConv2d(in_channels=3, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            RLNConv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                      stride=1, padding=0),
            RLNInstanceNorm2d(num_features=channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        ]
        pln_layers = [
                      OverLinear(in_features=representation_size,
                                out_features=hidden_dim),
                      nn.ReLU(inplace=True),
                      OverLinear(in_features=hidden_dim,
                                out_features=classes),
                      ]
        return nn.ModuleList(rln_layers), nn.ModuleList(pln_layers)



