import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Rev-in
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, in_channels, num_hidden, out_channels, drop_rate, if_ln=True):
        super(ResidualBlock, self).__init__()
        self.dense1 = nn.Linear(in_channels, num_hidden)
        self.dense2 = nn.Linear(num_hidden, out_channels)
        self.res_conv = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(p=drop_rate)
        self.ln = nn.LayerNorm(out_channels)
        self.if_ln = if_ln

    def forward(self, x):
        # dtbanuc covariate: [batch_size, in_out_lens, in_channels]
        # encoder: [batch, num_nodes, in_lens+num_static+(in_lens+out_lens)*r_hat]
        x_res = x
        x = F.relu(self.dense1(x))
        x = self.dropout(self.dense2(x))
        x_res = self.res_conv(x_res)
        output = x + x_res

        if self.if_ln:
            output = self.ln(output)

        return output


class FeatureProjection(nn.Module):
    def __init__(self, in_channels, num_hidden, out_channels, in_out_lens, drop_rate, if_fn=True):
        super(FeatureProjection, self).__init__()
        self.feature_projection_list = nn.ModuleList()
        self.in_out_lens = in_out_lens
        # for _ in range(in_out_lens):
        #     self.feature_projection_list.append(ResidualBlock(in_channels, num_hidden, out_channels, drop_rate, if_fn))
        self.feature_projection_list.append(ResidualBlock(in_channels, num_hidden, out_channels, drop_rate, if_fn))

    def forward(self, x):
        # [batch, in_out_lens, dynamic_covariate_dim]
        batch_size = x.size(0)
        # output_list = []
        # for i in range(self.in_out_lens):
        #     input = x[:, i, :]
        #     # output.shape [batch, r_hat]
        #     output = self.feature_projection_list[i](input)
        #     output_list.append(output)
        # # output.shape [batch, in_out_lens*r_hat]
        # output = torch.cat(output_list, dim=1)

        output = self.feature_projection_list[0](x).reshape(batch_size, -1)

        return output


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hidden, num_layers, drop_rate, if_fn=True):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList()
        if num_layers == 1:
            self.encoder_layers.append(ResidualBlock(in_channels, num_hidden, num_hidden, drop_rate, if_fn))
        else:
            self.encoder_layers.append(ResidualBlock(in_channels, num_hidden, num_hidden, drop_rate, if_fn))
            for _ in range(num_layers - 1):
                self.encoder_layers.append(ResidualBlock(num_hidden, num_hidden, num_hidden, drop_rate, if_fn))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        return x


class Decoder(nn.Module):
    def __init__(self,  num_hidden, out_channels, num_layers, drop_rate, if_fn=True):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList()
        if num_layers == 1:
            self.decoder_layers.append(ResidualBlock(num_hidden, num_hidden, out_channels, drop_rate, if_fn))
        else:
            for _ in range(num_layers - 1):
                self.decoder_layers.append(ResidualBlock(num_hidden, num_hidden, num_hidden, drop_rate, if_fn))
            self.decoder_layers.append(ResidualBlock(num_hidden, num_hidden, out_channels, drop_rate, if_fn))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x)

        return x


class TemporalDecoder(nn.Module):
    """
    TemporalDecoder
    in_channels: decoderOutputDim + r_hat
    """
    def __init__(self, in_channels, num_hidden, out_lens, drop_rate, if_fn=True):
        super(TemporalDecoder, self).__init__()
        self.temporal_decoder_list = nn.ModuleList()
        self.out_lens = out_lens
        # for _ in range(out_lens):
        #     self.temporal_decoder_list.append(ResidualBlock(in_channels, num_hidden, 1, drop_rate, if_fn))
        self.temporal_decoder_list.append(ResidualBlock(in_channels, num_hidden, 1, drop_rate, if_fn))

    def forward(self, x):
        # [batch, num_nodes, out_lens, decoderOutputDim + r_hat]
        assert self.out_lens == x.size(2)

        # output_list = []
        # for i in range(self.out_lens):
        #     input = x[:, :, i, :]
        #
        #     # output.shape [batch, num_nodes, 1]
        #     output = self.temporal_decoder_list[i](input)
        #     output_list.append(output)
        #
        # # output.shape [batch, num_nodes, out_lens]
        # output = torch.cat(output_list, dim=2)

        output = self.temporal_decoder_list[0](x)

        return output.squeeze(-1)

