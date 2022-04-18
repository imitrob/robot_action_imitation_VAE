# Base VAE class definition
import numpy as np
import torch, cv2
import torch.nn as nn
import torch.distributions as dist
from vae.models import encoders, decoders
from vae.utils import get_mean, kl_divergence, Constants, load_images, lengths_to_mask
from vae.vis import t_sne, tensors_to_df, plot_embeddings, plot_kls_df
from torch.utils.data import DataLoader
import pickle, os
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, enc, dec, feature_dim, enc_params, dec_params, n_latents, data_path=None, prior_dist=dist.Normal, likelihood_dist=dist.Normal, post_dist=dist.Normal):
        super(VAE, self).__init__()
        self.device = None
        self.pz = prior_dist
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0
        self.pth = data_path
        self.data_dim = feature_dim
        self.classes = nn.ParameterDict({"wave":torch.nn.Parameter(torch.tensor(0).float()), "dance":torch.nn.Parameter(torch.tensor(1).float()), "fly":torch.nn.Parameter(torch.tensor(2).float())})
        self.n_latents = n_latents
        self.enc_name, self.dec_name = enc, dec
        self.enc, self.dec = self.get_nework_classes(enc, dec)
        self.actionbias = nn.Parameter(torch.randn(3, n_latents))
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, n_latents), requires_grad=False)  # logvar
        ])
        self.modelName = 'vae_imit'
        #self.w2v = W2V(feature_dim, self.pth) if "word2vec" in self.pth else None

    def get_nework_classes(self, enc, dec):
       assert hasattr(encoders, "Enc_{}".format(enc)), "Did not find encoder {}".format(enc)
       enc_obj = getattr(encoders, "Enc_{}".format(enc))(self.n_latents, self.data_dim, **self.enc_params)
       assert hasattr(decoders, "Dec_{}".format(dec)), "Did not find decoder {}".format(enc)
       dec_obj = getattr(decoders, "Dec_{}".format(dec))(self.n_latents, self.data_dim, **self.dec_params)
       return enc_obj, dec_obj

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def getDataLoaders(self, batch_size,device="cuda"):
        self.device = device
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        if not ".pkl" in self.pth:
            if "image" in self.pth:
                d = load_images(self.pth, self.data_dim)
            else: raise Exception("If {} is an image dataset, please include 'image' in it's name. "
                                  "For other data types you should use .pkl or write your own dataLoader'".format(self.pth))
        else:
            with open(self.pth, 'rb') as handle:
                d = pickle.load(handle)
            d = [torch.from_numpy(np.asarray(x[0])) for x in d]
            if len(d[0].shape) < 3:
                    d = [torch.unsqueeze(i, dim=1) for i in d]
            kwargs["collate_fn"] = self.seq_collate_fn
        t_dataset = d[:int(len(d)*(0.9))]
        v_dataset = d[int(len(d)*(0.9)):]
        t_dataset = torch.utils.data.TensorDataset(torch.tensor(t_dataset))
        v_dataset = torch.utils.data.TensorDataset(torch.tensor(v_dataset))
        train = DataLoader(t_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test = DataLoader(v_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

    def forward(self, x, K=1):
        label = None
        if isinstance(x, list):
           label = torch.tensor(x[1])
           if len(label.unsqueeze(0)) == 1:
               try:
                label = int(label.unsqueeze(0))
               except:
                   label = [int(l) for l in list(label)]
           elif isinstance(label, int):
               label = label
           else:
               label = [int(l) for l in list(label)]
           x = x[0]
        mask = torch.tensor(np.ones((x.shape[0], x.shape[1]), dtype=bool)).cuda()
        for i1, v in enumerate(x):
            for i2, m in enumerate(v):
                if m.sum() == 0:
                    mask[i1][i2] = False
        self._qz_x_params = self.enc([x, mask.cuda()])
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        if label is not None:
               zs = zs + self.actionbias[label]
        zs = zs.reshape(x.shape[0]*K, -1)
        if K > 1:
            mask = mask.repeat(K, 1)
        px_z = self.px_z(*self.dec([zs, mask.cuda()]))
        return qz_x, px_z, zs

    def seq_collate_fn(self, batch):
        new_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in batch]))).to(self.device)
        return new_batch, masks

    def generate(self, runPath, epoch):
        N, K = 36, 1
        samples = self.generate_samples(N, K).cpu().squeeze()
        r_l = []
        if "image" in self.pth:
            for r, recons_list in enumerate(samples):
                    recon = recons_list.cpu().reshape(*self.data_dim).unsqueeze(0)
                    r_l = np.asarray(recon) if r_l == [] else np.concatenate((r_l, np.asarray(recon)))
            r_l = np.vstack((np.hstack(r_l[:6]), np.hstack(r_l[6:12]), np.hstack(r_l[12:18]), np.hstack(r_l[18:24]),  np.hstack(r_l[24:30]),  np.hstack(r_l[30:36])))
            cv2.imwrite('{}/visuals/gen_samples_{:03d}.png'.format(runPath, epoch), r_l*255)

    def reconstruct(self, data, labels, N=20):
        recons_mat = self.reconstruct_data(data[:N], labels)
        return recons_mat

    def analyse(self, data, runPath, epoch, labels=None):
        d = [torch.from_numpy(np.asarray(x)) for x in data]
        data_padded = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0)
        if len(data_padded.shape) < 4:
            data_padded = data_padded.unsqueeze(-1)
        zsl, kls_df = self.analyse_data(torch.tensor(data_padded), K=1, runPath=runPath, epoch=epoch, labels=labels)
        plot_kls_df(kls_df, '{}/visuals/kl_distance_{:03d}.png'.format(runPath, epoch))

    def generate_samples(self, N, label):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            if label is not None:
                 latents = latents + self.actionbias[label]
            px_z = self.px_z(*self.dec([latents, None]))
            data = get_mean(px_z)
        return data

    def reconstruct_data(self, data, labels):
        self.eval()
        d = [torch.from_numpy(np.asarray(x)) for x in data]
        d = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0)
        if labels is not None:
            d = [d, labels]
        output = self.forward(d)
        return output

    def analyse_data(self, data, K, runPath, epoch, labels=None):
        self.eval()
        if labels:
            data = [data, labels]
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, len(data)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        labels_indices = [list(self.classes.values()).index(c) for c in labels]
        labels_str = [list(self.classes.keys())[k] for k in labels_indices]
        t_sne([x.detach().cpu() for x in zss[1:]], runPath, epoch, K, labels_str)
        return torch.cat(zsl, 0).cpu().numpy(), kls_df
