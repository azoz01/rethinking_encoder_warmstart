import torch

from torch import nn


class Dataset2Vec(nn.Module):
    def __init__(
        self,
        activation=nn.ReLU(),
        f_dense_hidden_size=32,
        f_res_hidden_size=32,
        f_res_n_hidden=3,
        f_dense_out_hidden_size=32,
        f_block_repetitions=7,
        g_layers_sizes=[32, 16, 8],
        h_dense_hidden_size=32,
        h_res_hidden_size=32,
        h_res_n_hidden=3,
        h_dense_out_hidden_size=16,
        h_block_repetitions=3,
    ):
        super().__init__()

        f_components = [nn.Linear(2, f_dense_hidden_size)]
        for _ in range(f_block_repetitions):
            f_components.append(
                ResidualBlock(
                    input_size=f_dense_hidden_size,
                    units=f_res_hidden_size,
                    n_hidden=f_res_n_hidden,
                    output_size=f_dense_hidden_size,
                    activation=activation,
                )
            )
        f_components.append(nn.Linear(f_dense_hidden_size, f_dense_out_hidden_size))

        g_components = [nn.Linear(f_dense_out_hidden_size, g_layers_sizes[0])]
        for previous_layer_size, layer_size in zip(g_layers_sizes[:-1], g_layers_sizes[1:]):
            g_components.append(nn.Linear(previous_layer_size, layer_size))
            g_components.append(activation)

        h_components = [nn.Linear(g_layers_sizes[-1], h_dense_hidden_size)]
        for _ in range(h_block_repetitions):
            h_components.append(
                ResidualBlock(
                    input_size=h_dense_hidden_size,
                    units=h_res_hidden_size,
                    n_hidden=h_res_n_hidden,
                    output_size=h_dense_hidden_size,
                    activation=activation,
                )
            )
        h_components.append(nn.Linear(h_dense_hidden_size, h_dense_out_hidden_size))

        self.f = nn.Sequential(*f_components)
        self.g = nn.Sequential(*g_components)
        self.h = nn.Sequential(*h_components)

    def forward(self, X, y, *args, **kwargs):
        X_proc = X.T.repeat_interleave(y.shape[1], dim=0)
        y_proc = y.T.repeat(X.shape[1], 1)
        concatenated = torch.stack((X_proc, y_proc), 2)

        feat_target_means = self.f(concatenated).mean(dim=1)
        mean_from_all_feat_targets = self.g(feat_target_means).mean(dim=0)
        final_representation = self.h(mean_from_all_feat_targets)
        return final_representation


class FeedForward(nn.Module):
    def __init__(self, input_size, units, n_hidden, output_size, activation):
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.activation = activation

        components = [nn.Linear(self.input_size, self.units), activation]
        for _ in range(self.n_hidden - 1):
            components.append(nn.Linear(self.units, self.units))
            components.append(self.activation)
        components.append(nn.Linear(self.units, self.output_size))
        components.append(activation)

        self.block = nn.Sequential(*components)

    def forward(self, X, *args, **kwargs):
        return self.block(X)


class ResidualBlock(FeedForward):
    def __init__(self, input_size, units, n_hidden, output_size, activation):
        super().__init__(input_size, units, n_hidden, output_size, activation)

    def forward(self, X, *args, **kwargs):
        return X + super().forward(X)
