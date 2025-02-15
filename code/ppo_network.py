import torch
from torch import nn, optim
from torch.nn import functional as F


class PPONetworkBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PPONetworkBase, self).__init__()

        self.input_dims = kwargs.get("input_dims", None)

        self.shared_hidden_dims = kwargs.get("shared_hidden_dims")
        self.shared_norm = kwargs.get("shared_norm", None)
        self.shared_activation = kwargs.get("shared_activation", None)

        self.value_hidden_dims = kwargs.get("value_hidden_dims")
        self.value_norm = kwargs.get("value_norm", None)
        self.value_activation = kwargs.get("value_activation", None)

        self.output_dims = kwargs.get("output_dims", None)

        self.debug_prints = kwargs.get("debug_prints", False)

        assert self.input_dims != None, "Input dimensions must be provided"
        assert (
            self.shared_activation != None
        ), "Shared activation function must be provided"

        assert (
            self.value_activation != None
        ), "Value activation function must be provided"
        assert self.output_dims != None, "Output dimensions must be provided"

        # Shared layers
        self.shared_layers = self.build_layers(
            self.input_dims,
            self.shared_hidden_dims,
            self.shared_activation,
            self.shared_norm,
        )

        # Value head layers
        self.value_layers = self.build_layers(
            self.shared_hidden_dims[-1],
            self.value_hidden_dims,
            self.value_activation,
            self.value_norm,
        )
        self.value_output = nn.Linear(self.value_hidden_dims[-1], 1)

    def weight_initialization(self, layers):
        # Supported activations for He initialization
        he_supported_activations = (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.ELU)

        # TODO - Add support for other initialization methods

        # Apply He initialization to all Linear layers followed by supported activations
        for i, module in enumerate(layers):
            # Check if the current module is Linear and is followed by a supported activation
            if isinstance(module, nn.Linear):
                next_module = layers[i + 1] if (i + 1) < len(layers) else None
                if isinstance(next_module, he_supported_activations):
                    if self.debug_prints:
                        print(f"He initialization applied to layer {i}")
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def build_layers(self, input_size, layer_dims, activation, normalize=None):
        # Dynamic network builder
        layers = []
        for dim in layer_dims:
            layers.append(nn.Linear(input_size, dim))
            if normalize != None:
                layers.append(normalize(dim))
            layers.append(activation())
            input_size = dim

        # Apply weight initialization
        self.weight_initialization(layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError  # Must be implemented in subclass


class PPONetworkDiscrete(PPONetworkBase):
    def __init__(self, *args, **kwargs):
        super(PPONetworkDiscrete, self).__init__(*args, **kwargs)

        self.policy_hidden_dims = kwargs.get("policy_hidden_dims")
        self.policy_norm = kwargs.get("policy_norm", None)
        self.policy_activation = kwargs.get("policy_activation", None)

        assert (
            self.policy_activation != None
        ), "Policy activation function must be provided"

        # Policy head layers
        self.policy_layers = self.build_layers(
            self.shared_hidden_dims[-1],
            self.policy_hidden_dims,
            self.policy_activation,
            self.policy_norm,
        )
        self.policy_output = nn.Linear(self.policy_hidden_dims[-1], self.output_dims)

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


class PPONetworkContinuous(PPONetworkBase):
    def __init__(self, *args, **kwargs):
        super(PPONetworkContinuous, self).__init__(*args, **kwargs)

        self.mean_hidden_dims = kwargs.get("mean_hidden_dims")
        self.mean_norm = kwargs.get("mean_norm", None)
        self.mean_activation = kwargs.get("mean_activation", None)

        self.log_var_hidden_dims = kwargs.get("log_var_hidden_dims")
        self.log_var_norm = kwargs.get("log_var_norm", None)
        self.log_var_activation = kwargs.get("log_var_activation", None)

        assert self.mean_activation != None, "Mean activation function must be provided"
        assert (
            self.log_var_activation != None
        ), "Log variance activation function must be provided"

        # Mean head layers
        self.mean_layers = self.build_layers(
            self.shared_hidden_dims[-1],
            self.mean_hidden_dims,
            self.mean_activation,
            self.mean_norm,
        )
        self.mean_output = nn.Linear(self.mean_hidden_dims[-1], self.output_dims)

        # Log variance head layers
        self.log_var_layers = self.build_layers(
            self.shared_hidden_dims[-1],
            self.log_var_hidden_dims,
            self.log_var_activation,
            self.log_var_norm,
        )
        self.log_var_output = nn.Linear(self.log_var_hidden_dims[-1], self.output_dims)

    def forward(self, x):
        # Pass through shared layers
        x = self.shared_layers(x)

        # Mean head
        mean_x = self.mean_layers(x)
        mean = self.mean_output(mean_x)

        # Log variance head
        log_var_x = self.log_var_layers(x)
        log_var = self.log_var_output(log_var_x)

        value_x = self.value_layers(x)
        value = self.value_output(value_x)

        return mean, log_var, value
