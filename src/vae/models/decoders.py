import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vae.models.nn_modules import PositionalEncoding

class Dec_Transformer(nn.Module):
    def __init__(self, latent_dim, data_dim=1, ff_size=1024, num_layers=1, num_heads=4, dropout=0.1, activation="gelu"):
        super(Dec_Transformer, self).__init__()
        self.name = "Transformer"
        self.njoints = data_dim[1]
        self.nfeats = data_dim[2]
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.sequence_pos_encoder = torch.nn.DataParallel(PositionalEncoding(self.latent_dim, self.dropout))

        seqTransDecoderLayer = torch.nn.DataParallel(nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation))
        self.seqTransDecoder = torch.nn.DataParallel(nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers))
        self.finallayer = torch.nn.DataParallel(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, batch):
        z = batch[0]
        mask = batch[1]
        latent_dim = z.shape[-1]
        bs = z.shape[0]
        mask = torch.tensor(np.ones((bs,self.data_dim[0]), dtype=bool)).to(z.device) if mask is None else mask
        timequeries = torch.zeros(mask.shape[1], bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(mask.shape[1], bs, self.njoints,  self.nfeats)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2, 3)
        return output.to(z.device), torch.tensor(0.75).to(z.device)
