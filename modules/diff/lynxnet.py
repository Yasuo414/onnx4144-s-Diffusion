import torch
import math

from utils.hparams import hparams

class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
    
    def forward(self, x):
        device = x.device
        half_dimension = self.dimension // 2
        embedding = math.log(10000) / (half_dimension - 1)
        embedding = torch.exp(torch.arange(half_dimension, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding
    
class SwiGLU(torch.nn.Module):
    def __init__(self, dimension=-1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        output, gate = x.chunk(2, dim=self.dimension)

        return output * torch.nn.functional.silu(gate)
    
class Conv1d(torch.nn.Conv1d):
    def __init__(self, *arguments, **kwargs):
        super().__init__(*arguments, **kwargs)
        torch.nn.init.kaiming_normal_(self.weight)

class Transpose(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        assert len(dimensions) == 2, "Dimensions must be a tuple of two dimensions"
        self.dimensions = dimensions
    
    def forward(self, x):
        return x.transpose(*self.dimensions)
    
class ConvModule(torch.nn.Module):
    @staticmethod
    def calculate_same_padding(kernel_size):
        padding = kernel_size // 2

        return padding, padding - (kernel_size + 1) % 2
    
    def __init__(self, dimension, expansion_factor, kernel_size=31, activation="PReLU", dropout=0.0):
        super().__init__()
        inner_dimension = dimension * expansion_factor

        activation_classes = {
            "SiLU": torch.nn.SiLU(),
            "ReLU": torch.nn.ReLU(),
            "PReLU": lambda: torch.nn.PReLU(inner_dimension)
        }

        activation = activation if activation is not None else "PReLU"
        if activation not in activation_classes:
            raise ValueError(f"Activation '{activation}' is not a valid activation")
        
        _activation = activation_classes[activation]()
        padding = self.calculate_same_padding(kernel_size)

        if float(dropout) > 0.:
            _dropout = torch.nn.Dropout(dropout)
        else:
            _dropout = torch.nn.Identity()
        
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(dimension),
            Transpose((1, 2)),
            torch.nn.Conv1d(dimension, inner_dimension * 2, 1),
            SwiGLU(dimension=1),
            torch.nn.Conv1d(inner_dimension, inner_dimension, kernel_size=kernel_size, padding=padding[0], groups=inner_dimension),
            _activation,
            torch.nn.Conv1d(inner_dimension, dimension, 1),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.network(x)
    
class ResidualLayer(torch.nn.Module):
    def __init__(self, dimension_condition, dimension, expansion_factor, kernel_size=31, activation="PReLU", dropout=0.0):
        super().__init__()
        self.diffusion_projection = torch.nn.Conv1d(dimension, dimension, 1)
        self.conditioner_projection = torch.nn.Conv1d(dimension_condition, dimension, 1)
        self.conv_module = ConvModule(dimension, expansion_factor, kernel_size, activation, dropout)
    
    def forward(self, x, conditioner, step, front_condition_inject=False):
        if front_condition_inject:
            x = x + self.conditioner_projection(conditioner)
            residual_x = x
        else:
            residual_x = x
            x = x + self.conditioner_projection(conditioner)
        
        x = x + self.diffusion_projection(step)
        x = x.transpose(1, 2)
        x = self.conv_module(x)
        x = x.transpose(1, 2) + residual_x

        return x

class LYNXNet(torch.nn.Module):
    def __init__(self, in_dims=128, num_features=1, num_layers=20, num_channels=512, expansion_factor=2, kernel_size=31, activation="PReLU", dropout=0.0, strong_condition=False):
        super().__init__()
        self.in_dimension = in_dims
        self.num_features = num_features
        self.condition_dimension = hparams["hidden_size"]
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.expansion_factor = expansion_factor
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.strong_condition = strong_condition

        self.input_projection = Conv1d(in_dims * num_features, num_channels, 1)

        self.diffusion_embedding = torch.nn.Sequential(
            SinusoidalPositionalEmbedding(num_channels),
            torch.nn.Linear(num_channels, num_channels * 4),
            torch.nn.GELU(),
            torch.nn.Linear(num_channels * 4, num_channels),
        )

        self.residual_layers = torch.nn.ModuleList([
            ResidualLayer(self.condition_dimension, num_channels, expansion_factor, kernel_size, activation, dropout)
            for _ in range(num_layers)
        ])

        self.normalize = torch.nn.LayerNorm(num_channels)
        output_projection_layer = Conv1d(num_channels, in_dims * num_features, kernel_size=1)

        if hparams["diff_type"] == "ddpm":
            torch.nn.init.zeros_(output_projection_layer.weight)
        
        output_layers = [output_projection_layer]
        if hparams["diff_type"] == "reflow":
            output_layers.append(torch.nn.Tanh())
        
        self.output_projection = torch.nn.Sequential(*output_layers)

    def forward(self, spec, diffusion_step, cond, cond_mask_ratio=0.0):
        use_4_dimensional_output = False
        if spec.dim() == 4:
            use_4_dimensional_output = True

            if self.num_features == 1:
                x = spec[:, 0]
            else:
                x = spec.flatten(start_dim=1, end_dim=2)
        else:
            x = spec

        x = self.input_projection(x)

        if not self.strong_condition:
            x = torch.nn.functional.gelu(x)
        
        if diffusion_step.dim() == 2:
            diffusion_step = diffusion_step.squeeze(-1)
        elif diffusion_step.dim() == 1:
            pass
        
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)

        if self.training and cond_mask_ratio > 0:
            mask = torch.bernoulli(torch.ones(cond.shape[0], device=cond.device) * cond_mask_ratio).view(-1, 1, 1)
            cond = cond * (1.0 - mask)
        
        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step, front_condition_inject=self.strong_condition)
        
        x = self.normalize(x.transpose(1, 2)).transpose(1, 2)
        x = self.output_projection(x)

        if use_4_dimensional_output:
            if self.num_features == 1:
                x = x[:, None, :, :]
            else:
                x = x.reshape(-1, self.num_features, self.in_dimension, x.shape[2])
        
        return x