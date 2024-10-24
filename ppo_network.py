import torch
from torch import nn, optim

class PPONetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PPONetwork, self).__init__()
        self.input_dim = kwargs.get("input_dim")

        self.shared_hidden_dims = kwargs.get("shared_hidden_dims")
        self.shared_norm = kwargs.get("shared_norm", None)

        self.policy_hidden_dims = kwargs.get("policy_hidden_dims")
        self.policy_norm = kwargs.get("policy_norm", None)

        self.value_hidden_dims = kwargs.get("value_hidden_dims")
        self.value_norm = kwargs.get("value_norm", None)

        self.output_dim = kwargs.get("output_dim")

        # Shared layers
        self.shared_layers = self.build_layers(
            self.input_dim, self.hidden_dims, self.shared_norm
        )

        # Policy head layers
        self.policy_layers = self.build_layers(
            self.hidden_dims[-1], self.policy_hidden_dims, self.policy_norm
        )
        self.policy_output = nn.Linear(self.policy_hidden_dims[-1], self.output_dim)

        # Value head layers
        self.value_layers = self.build_layers(
            self.hidden_dims[-1], self.value_hidden_dims, self.value_norm
        )
        self.value_output = nn.Linear(self.value_hidden_dims[-1], 1)

        # Apply He initialization to the layers
        self.apply(self.he_initialization)

    def build_layers(self, input_size, layer_dims, normalize=None):
        layers = []
        for dim in layer_dims:
            layers.append(nn.Linear(input_size, dim))
            if normalize != None:
                layers.append(normalize(dim))
            layers.append(nn.ReLU())
            input_size = dim
        return nn.Sequential(*layers)

    def he_initialization(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Pass through shared layers
        x = self.shared_layers(x)

        # Policy head
        policy_x = self.policy_layers(x)
        policy = F.softmax(self.policy_output(policy_x), dim=-1)

        # Value head
        value_x = self.value_layers(x)
        value = self.value_output(value_x)

        return policy, value
