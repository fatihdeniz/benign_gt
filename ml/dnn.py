import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc_layers[0](x))

        for i in range(1, self.num_layers):
            x = F.relu(self.fc_layers[i](x))

        x = self.output_layer(x)
        return x
