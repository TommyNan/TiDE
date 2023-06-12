import torch
import torch.nn as nn
from .module import FeatureProjection, Encoder, Decoder, TemporalDecoder, RevIN


class TiDE(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, **configs):
        super(TiDE, self).__init__()
        model_configs = configs['model']
        self.in_lens = model_configs['in_lens']
        self.out_lens = model_configs['out_lens']
        self.num_nodes = model_configs['num_nodes']
        self.num_hidden = model_configs['num_hidden']
        self.num_hidden_covar = 64
        self.num_hidden_temp_dec = model_configs['num_hidden_temp_dec']
        self.in_channels_covar = model_configs['in_channels_covar']
        self.out_channels_covar = 4
        self.num_hidden_static = 16
        self.out_channels_dec = model_configs['out_channels_dec']
        self.num_layers_enc = model_configs['num_layers_enc']
        self.num_layers_dec = model_configs['num_layers_dec']
        self.if_ln = model_configs['if_ln']
        self.if_revin = model_configs['if_revin']
        self.drop_rate = model_configs['drop_rate']
        self.in_out_lens = self.in_lens + self.out_lens

        self.revin_layer = RevIN(num_features=self.num_nodes)
        self.feature_projection = FeatureProjection(in_channels=self.in_channels_covar,
                                                    num_hidden=self.num_hidden_covar,
                                                    out_channels=self.out_channels_covar,
                                                    in_out_lens=self.in_out_lens,
                                                    drop_rate=self.drop_rate,
                                                    if_fn=self.if_ln,
                                                    )

        self.encoder = Encoder(
                               in_channels=self.in_lens+self.num_hidden_static+self.in_out_lens*self.out_channels_covar,
                               # in_channels=self.in_lens+self.in_out_lens*self.out_channels_covar,
                               num_hidden=self.num_hidden,
                               num_layers=self.num_layers_enc,
                               drop_rate=self.drop_rate,
                               if_fn=self.if_ln,
                               )

        self.decoder = Decoder(num_hidden=self.num_hidden,
                               out_channels=self.out_lens*self.out_channels_dec,
                               num_layers=self.num_layers_dec,
                               drop_rate=self.drop_rate,
                               if_fn=self.if_ln,
                               )

        self.temp_decoder = TemporalDecoder(in_channels=self.out_channels_covar+self.out_channels_dec,
                                            num_hidden=self.num_hidden_temp_dec,
                                            out_lens=self.out_lens,
                                            drop_rate=self.drop_rate,
                                            if_fn=self.if_ln,
                                            )
        self.res_conv = nn.Linear(self.in_lens, self.out_lens)
        self.attr_embedding = nn.Embedding(self.num_nodes, self.num_hidden_static)

    def forward(self, x, x_timestamp):
        # x.shape [batch_size, in_lens, num_nodes(in_channels)] x_timestamp.shape [batch_size, in_out_lens, num_features]

        batch_size, in_lens, num_nodes = x.shape

        if self.if_revin:
            x = self.revin_layer(x, 'norm')

        # x_lookback [batch_size, in_lens, num_nodes] -> [batch, num_nodes, in_lens]
        x_lookback = x.permute(0, 2, 1)

        # Attributes [batch_size, num_nodes, num_static]
        nodes_index = torch.arange(num_nodes).to(x.device)
        x_attr = self.attr_embedding(nodes_index)
        x_attr = x_attr.unsqueeze(0).expand([batch_size, -1, -1]).contiguous()

        # Dynamic Covariate [batch_size, in_out_lens, num_covar]->[batch_size, num_nodes, in_out_lens * temporalWidth(4)]
        x_covar = self.feature_projection(x_timestamp)
        x_covar = x_covar.unsqueeze(1).expand(-1, num_nodes, -1).contiguous()

        # [batch, num_nodes, in_lens+1+in_out_lens*4]->[batch, num_nodes, num_hidden]
        # ->[batch, num_nodes, self.out_lens*self.out_channels_dec]
        x_enc = torch.cat([x_lookback, x_attr, x_covar], dim=2)
        # x_enc = torch.cat([x_lookback, x_covar], dim=2)
        x_e = self.encoder(x_enc)
        x_g = self.decoder(x_e)
        x_g = x_g.view(batch_size, num_nodes, self.out_lens, self.out_channels_dec).contiguous()

        x_covar_out = x_covar[:, :, -self.out_lens*self.out_channels_covar:]
        x_covar_out = x_covar_out.view(batch_size, num_nodes, self.out_lens, self.out_channels_covar).contiguous()

        # x_dec [batch, num_nodes, out_lens, decoderOutputDim + r_hat]->[batch, num_nodes, out_lens]
        x_dec = torch.cat([x_g, x_covar_out], dim=3)
        x_dec = self.temp_decoder(x_dec)
        x_res = self.res_conv(x_lookback)

        y_pred = x_dec + x_res
        y_pred = y_pred.permute(0, 2, 1)

        if self.if_revin:
            y_pred = self.revin_layer(y_pred, 'denorm')

        return y_pred # y_pred.shape [batch_size, out_lens, num_nodes]







