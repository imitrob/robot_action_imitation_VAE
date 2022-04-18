import json, yaml
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch import optim
import os, glob, random
import vae.models as models
import vae.objectives as objectives
from vae.utils import Logger, Timer, save_model, save_vars, check_action_ok, label_to_cls
from vae.utils import Adabelief


def pack_data(data):
    d = [torch.from_numpy(np.asarray(x)) for x in data]
    data = torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0).unsqueeze(-1)
    return data

def pick_single(what, size=8):
    new_d = []
    for i, x in enumerate(what):
        new_m = [x[:size]/180]
        new_d.append(new_m)
    return new_d

def load_actions_from_path(path, sort=True):
    all_files = glob.glob(os.path.join(path, "./*/**.pkl"))
    data = []
    labels = []
    for f in all_files:
        labels.append(os.path.basename(os.path.dirname(f)).split("_")[0])
    if sort:
        all_files = [x for _, x in sorted(zip(labels, all_files))]
        labels = sorted(labels)
    for f in all_files:
        with open(f, 'rb') as handle:
             poses = pickle.load(handle)
             data.append(poses)
    if not sort:
        data = list(zip(data, labels))
        random.shuffle(data)
        data, labels = zip(*data)
    return data, labels

def get_loss_mean(loss):
    return round(float(torch.mean(torch.tensor(loss).detach().cpu())),3)

def pad_seq_data(data, masks):
    for i, _ in enumerate(data):
        if masks[i] is not None:
            data[i].append(masks[i])
        else:
            data[i] = [torch.tensor(o[0]) for o in data[i][0]]
    return data

class IncremVAE():
    def __init__(self, cfg_pth, cuda=True):
        self.config, self.mods = self.parse_args(cfg_pth)
        self.device = None
        self.model = None
        self.runPath = None
        self.optim = None
        self.batch_size = None
        self.objective = None
        self.K = None
        self.actionlen = None
        self.testdata = None
        self.setup_model(cuda)
        self.iternum = 0
        self.lossmeter = Logger(self.runPath, self.mods)
        self.agg = defaultdict(list)
        self.data_container = [[],[]]

    def parse_args(self, cfg):
        with open(cfg) as file: config = yaml.load(file)
        modalities = []
        for x in range(20):
            if "modality_{}".format(x) in list(config.keys()):
                modalities.append(config["modality_{}".format(x)])
        return config, modalities

    def setup_model(self, cuda):
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.backends.cudnn.benchmark = True
        cuda = True if (torch.cuda.is_available() and cuda) else False
        self.device = torch.device("cuda" if cuda else "cpu")

        model = "VAE" if len(self.mods) == 1 else self.config["mixing"].lower()
        modelC = getattr(models, model)
        params = [[m["encoder"] for m in self.mods], [m["decoder"] for m in self.mods], [m["feature_dim"] for m in self.mods]]
        if len(self.mods) == 1:
            params = [x[0] for x in params]
        enc_params = {"ff_size":int(self.config["enc_ff"]), "num_layers":int(self.config["enc_layers"]), "num_heads":int(self.config["enc_heads"])}
        dec_params = {"ff_size":int(self.config["dec_ff"]), "num_layers":int(self.config["dec_layers"]), "num_heads":int(self.config["dec_heads"])}
        params.append(enc_params)
        params.append(dec_params)
        self.model = modelC(*params, self.config["n_latents"]).to(self.device)
        self.K = self.config["k"]
        if self.config["pre_trained"]:
            print('Loading model {} from {}'.format(model.modelName, self.config["pre_trained"]))
            self.model.load_state_dict(torch.load(self.config["pre_trained"] + '/model.rar'))
            self.model._pz_params = model._pz_params

        # set up run path
        self.runPath = os.path.join('results/', self.config["exp_name"])
        os.makedirs(self.runPath, exist_ok=True)
        os.makedirs(os.path.join(self.runPath, "visuals"), exist_ok=True)
        print('Expt:', self.runPath)

        # save args to run
        with open('{}/config.json'.format(self.runPath), 'w') as fp:
            json.dump(self.config, fp)

        self.actionlen = int(self.config["modality_1"]["feature_dim"][-1])
        self.testdata = load_actions_from_path(self.config["modality_1"]["testdata"], sort=True)
        self.batch_size = self.config["batch_size"]
        # preparation for training
        if "optimizer" not in self.config.keys() or self.config["optimizer"].lower() == "adam":
            self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(self.config["lr"]), amsgrad=True)
        elif self.config["optimizer"].lower() == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=0.00001, momentum=0.9)
        else:
            self.optim = Adabelief(self.model.parameters(), lr=float(self.config["lr"]), eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                      rectify=True, print_change_log=False)
        self.objective = getattr(objectives,
                            ('m_' if hasattr(model, 'vaes') else '')
                            + ("_".join((self.config["obj"], self.config["mixing"])) if hasattr(model, 'vaes') else self.config["obj"]))

    def pick_single(self, what):
        return pick_single(what, size=self.actionlen)

    def estimate_log_marginal(self, K=100):
        self.model.eval()
        data = pack_data([torch.from_numpy(np.asarray(self.pick_single(x))) for x in self.testdata[0]]).squeeze(-1)
        labels = [label_to_cls(self.model, l) for l in self.testdata[1]]
        return self.objective(self.model, [data, labels], K=K, ltype="lprob")[0]

    def train_step(self, dataT, label=None, replay=False):
        data = torch.tensor(dataT) if not torch.is_tensor(dataT) else dataT
        #self.data_container[0].append(data)
        #if label:
        #    self.data_container[1].append(label)
        self.model.train()
        loss_m = []
        kld_m = []
        partial_losses = [[] for _ in range(len(self.mods))]
        self.optim.zero_grad()
        if label:
            labels = [label_to_cls(self.model, l) for l in label]
            data = [data, labels]
        loss, kld, partial_l, _ = self.objective(self.model, data, K=self.K, ltype=self.config["loss"])
        loss_m.append(loss)
        kld_m.append(kld)
        for i, l in enumerate(partial_l):
            partial_losses[i].append(l)
        loss.backward()
        self.optim.step()
        if not replay:
            self.iternum += 1
            print('====> Iteration Step Train loss: {:.4f}'.format(loss))
            save_model(self.model, self.runPath + '/model.rar')
            if self.iternum % self.config["viz_freq"] == 0:
                os.makedirs(self.runPath + '/{}/'.format(self.iternum), exist_ok=True)
                with open('{}/config.json'.format(self.runPath + '/{}/'.format(self.iternum)), 'w') as fp:
                    json.dump(self.config, fp)
                save_model(self.model, self.runPath + '/{}/model.rar'.format(self.iternum))
                progress_d = {"Epoch": self.iternum, "Train Loss": get_loss_mean(loss_m),
                              "Train KLD": get_loss_mean(kld_m)}
                for i, x in enumerate(partial_losses):
                    progress_d["Train Mod_{}".format(i)] = get_loss_mean(x)
                self.lossmeter.update_train(progress_d)
                self.eval()
                self.eval_progress()

    def save_model(self, tag="model"):
        save_model(self.model, self.runPath + '/{}.rar'.format(tag))

    def eval(self, dataT=None, labels=None):
        self.model.eval()
        with torch.no_grad():
            if not dataT:
                splits, splitlabels = [], []
                data = pack_data([torch.from_numpy(np.asarray(self.pick_single(x))) for x in self.testdata[0]]).squeeze(-1)
                for val in ["dance", "wave", "fly"]:
                    indices = [i for i, j in enumerate(self.testdata[1]) if j == val]
                    splits.append(data[indices])
                    splitlabels.append([val] * len(indices))
            else:
                data = torch.tensor(dataT).unsqueeze(-1).unsqueeze(0)
            progress_d = {"Epoch": self.iternum}
            metrices = [{},{}, {}]
            for idx, split in enumerate(splits):
                labels = [label_to_cls(self.model, l) for l in splitlabels[idx]]
                data = [split, labels]
                loss_m = []
                kld_m = []
                partial_losses = [[] for _ in range(len(self.mods))]
                loss, kld, partial_l, metrics = self.objective(self.model, data, K=self.K, ltype=self.config["loss"], eval=True)
                for m, v in enumerate(metrics):
                    metrices[m][splitlabels[idx][0]] = v
                loss_m.append(loss)
                kld_m.append(kld)
                for i, l in enumerate(partial_l):
                    partial_losses[i].append(l)
                progress_d["Test Loss {}".format(splitlabels[idx][0])] = float(loss_m[0] / len(data[0]))
            self.save_metrics(metrices, type="recons")
            progress_d["Log Likelihood"] = float(self.estimate_log_marginal())
            self.lossmeter.update(progress_d)
            print('====>             Test loss: {:.4f}'.format(float(loss_m[0] / len(data[0]))))

    def eval_progress(self):
        self.model.eval()
        success_measures = [{}, {}, {}]
        for l in list(self.model.classes.keys()):
            sample = self.model.generate_samples(100, int(self.model.classes[l])).squeeze()
            for i, d in enumerate(["normal", "easy", "hard"]):
                success_matrix = check_action_ok(sample, [l]*len(sample), difficulty=d)
                success_measures[i][l] = sum(success_matrix) / len(success_matrix )
        self.save_metrics(success_measures)

    def save_metrics(self, metrics, type="samples"):
        type = "success_metrics_" if type == "samples" else "success_recons_"
        with open(os.path.join(self.runPath, "visuals/{}{}.txt".format(type, self.iternum)), "w") as text_file:
            for k in list(self.model.classes.keys()):
                text_file.write("Successful {}: {}\n".format(k, metrics[0][k]))
                text_file.write("Easy {}: {}\n".format(k, metrics[1][k]))
                text_file.write("Hard {}: {}\n".format(k, metrics[2][k]))

def detach(listtorch):
    return [np.asarray(l.detach().cpu()) for l in listtorch]


if __name__ == '__main__':
    pass
