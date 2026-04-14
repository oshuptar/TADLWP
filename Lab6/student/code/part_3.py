# part_3.py
import torch
import torch.nn as nn
from torchvision import models

from helpers.training_utils import get_dataloaders, visualize_experiment
from part_1 import ClassifierHead, train_model
from part_2 import SimpleResNeXt
from helpers.draw_architecture_helper import print_and_draw_model_structure

"""
Part 3 — Simple transfer learning example.

Uses:
- train_model from part_1
- SimpleResNeXt from part_2 as a small baseline
- pretrained ResNet18 as the transfer learning model
"""

class TransferResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        """
        TODO: Create a TransferResNet18 model with a pretrained ResNet18 backbone and a new classifier head.
        """
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights) ## a;ternatively test: resnet18(pretrained=True)

        # freezes all pretrained parameters
        for param in self.backbone.parameters():
            param.requires_grad_ = False

        # replaces final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        """
        TODO: Pass input through the TransferResNet18 model.
        """
        return self.backbone(x)


def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    histories = {
        "SimpleResNeXt": {"model": [], "history": []},
        "TransferResNet18": {"model": [], "history": []},
    }

    # baseline model from part_2
    baseline_model = SimpleResNeXt(num_classes=10, cardinality=4)

    trained_baseline, baseline_history = train_model(
        baseline_model,
        train_loader,
        val_loader,
        lr=0.01,
        criterion=nn.CrossEntropyLoss(),
        momentum=0.9,
        weight_decay=1e-4,
        epochs=8,
    )

    histories["SimpleResNeXt"]["model"].append(trained_baseline)
    histories["SimpleResNeXt"]["history"].append(baseline_history)

    # transfer learning model
    transfer_model = TransferResNet18(num_classes=10)

    trained_transfer, transfer_history = train_model(
        transfer_model,
        train_loader,
        val_loader,
        lr=0.01,
        criterion=nn.CrossEntropyLoss(),
        momentum=0.9,
        weight_decay=1e-4,
        epochs=8,
    )

    histories["TransferResNet18"]["model"].append(trained_transfer)
    histories["TransferResNet18"]["history"].append(transfer_history)

    visualize_experiment(histories)

    print_and_draw_model_structure(
        transfer_model,
        output_file=f"TransferResNet18_fx_graph",
        fmt="svg")

if __name__ == "__main__":
    main()