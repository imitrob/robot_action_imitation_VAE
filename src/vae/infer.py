import torch
import vae.models as models
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import glob

def parse_args(cfg_path=None):
    with open(os.path.join(os.path.dirname(cfg_path), "config.json")) as file: config = json.load(file)
    modalities = []
    for x in range(20):
        if "modality_{}".format(x) in list(config.keys()):
            modalities.append(config["modality_{}".format(x)])
    return config, modalities
plot_colors = ["blue", "green", "red", "cyan", "magenta", "orange", "navy", "maroon", "brown"]


def plot_setup(xname, yname, pth, figname):
    plt.xlabel(xname, fontsize=18)
    plt.ylabel(yname, fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    p = pth if os.path.isdir(pth) else os.path.dirname(pth)
    plt.savefig(os.path.join(p, "visuals/{}.png".format(figname)))
    plt.clf()

def plot_loss_action(path, vae=None):
    pth = os.path.join(path, "loss.csv") if not "loss.csv" in path else path
    loss = pd.read_csv(pth, delimiter=",")
    epochs = loss["Epoch"]
    l1 = loss["Test Loss wave"]
    l2 = loss["Test Loss dance"]
    l3 = loss["Test Loss fly"]
    likelihood = loss["Log Likelihood"]
    if vae:
        e = int(vae.config["epochs"])
        epochs = [int(i/e)*int(vae.config["batch_size"]) for i in list(loss["Epoch"])]
        l1 = loss["Test Loss wave"]
        l2 = loss["Test Loss dance"]
        l3 = loss["Test Loss fly"]
    plt.plot(epochs, l2, color='green', linestyle='solid', label="dance")
    plt.plot(epochs, l3, color='red', linestyle='solid', label="fly")
    plt.plot(epochs, l1, color='blue', linestyle='solid', label="wave")
    plot_setup("Training samples", "Loss", path, "loss")
    plt.plot(epochs, likelihood, color='black', linestyle='solid')
    plt.locator_params(axis="x", integer=True, tight=True)
    plot_setup("Training samples", "Log Likelihood", path, "likelihood")
    print("Saved loss plot")

def plot_metrics(paths, all=True, vae=None):
    if not isinstance(paths, list):
        paths = [paths]
        vae = [vae] if vae is not None else [None]
    else:
        all = False
    for pathind, path in enumerate(paths):
        all_files1 = [x for x in glob.glob(os.path.join(path, "./*/**.txt")) if "metrics" in x]
        all_files2 = [x for x in glob.glob(os.path.join(path, "./*/**.txt")) if "recons" in x]
        epochs = None
        if vae:
            vae = vae[pathind]
            epochs = int(vae.config["epochs"])
        for fileind, all_files in enumerate([all_files1, all_files2]):
            labels = [x.split("_")[-1].split(".txt")[0] for x in all_files]
            labels = [int(i) for i in labels]
            all_files = [x for _, x in sorted(zip(labels, all_files))]
            normal = [[],[],[]]
            easy = [[],[],[]]
            hard = [[],[],[]]
            for m in all_files:
                with open(m, 'r') as file:
                    txt = file.read().replace('\n', ' ').lower().split(" ")
                    normal[0].append(float(txt[2]))
                    flyn = float(float(txt[11]))
                    normal[1].append(flyn)
                    normal[2].append(float(txt[20]))
                    easy[0].append(float(txt[5]))
                    flye = float(float(txt[14]))
                    easy[1].append(flye)
                    easy[2].append(float(txt[23]))
                    hard[0].append(float(txt[8]))
                    flyh = float(float(txt[17]))
                    hard[1].append(flyh)
                    hard[2].append(float(txt[26]))
            names = [txt[1].split(":")[0], txt[10].split(":")[0], txt[19].split(":")[0]]
            colors = [["darkgreen", "red", "blue"], ["limegreen", "lightcoral", "lightskyblue"], ["limegreen", "lightcoral", "lightskyblue"]]
            diffs = ["(easy)", "(medium)", "(difficult)"]
            lines = ["solid",  "dashed", "dotted"]
            ls = [2, 1, 1]
            to_plot = [easy, normal, hard] if all else [normal]
            x = sorted(labels) if not epochs else [int(i/epochs)*int(vae.config["batch_size"]) for i in sorted(labels)]
            if fileind == 2:
                x = sorted(labels)
            for ind, data in enumerate(to_plot):
                for i, d in enumerate(data):
                    plt.plot(x, [s*100 for s in d], color=colors[pathind + ind][i], linestyle=lines[pathind + ind], linewidth=ls[pathind + ind], label=" ".join((names[i], diffs[ind])))
            tag = ["generated", "reconstructed"][fileind]
            plt.locator_params(axis="x", integer=True, tight=True)
            plot_setup("Training samples", "Valid generated samples (%)", path, "metrics_{}".format(tag))
        print("Saved loss plot")


def load_model(path, trainloader=None, testloader=None, batch=8):
    device = torch.device("cuda")
    pc = path if not ".rar" in path else os.path.dirname(path)
    config, mods = parse_args(cfg_path=os.path.join(pc + "/config.json"))
    model = "VAE" if len(mods) == 1 else config["mixing"].lower()
    modelC = getattr(models, model)
    params = [[m["encoder"] for m in mods], [m["decoder"] for m in mods], [m["path"] for m in mods],
              [m["feature_dim"] for m in mods]]
    if len(mods) == 1:
        params = [x[0] for x in params]
    enc_params = {"ff_size": int(config["enc_ff"]), "num_layers": int(config["enc_layers"]),
                  "num_heads": int(config["enc_heads"])}
    dec_params = {"ff_size": int(config["dec_ff"]), "num_layers": int(config["dec_layers"]),
                  "num_heads": int(config["dec_heads"])}
    model = modelC(enc=params[0],dec=params[1], feature_dim=params[-1],enc_params=enc_params, dec_params=dec_params, n_latents=config["n_latents"]).to(device)
    print('Loading model {} from {}'.format(model.modelName, path))
    p = path if "model" in path else os.path.join(path, "model.rar")
    model.load_state_dict(torch.load(p))
    model._pz_params = model._pz_params
    model.eval()
    return model
