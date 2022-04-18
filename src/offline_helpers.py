from vae.infer import load_model
import numpy as np
import torch, math, random
from vae.vae_learner import load_actions_from_path, pick_single
from vae.utils import check_action_ok, mean, stdev
from vae.infer import plot_loss_action, plot_metrics, load_model
import os
from matplotlib import pyplot as plt

def plot_hist(d, w, f, title, figname):
    plt.hist([d, w, f], label=['dance', 'fly', 'wave'], bins=50)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig("{}.png".format(figname))
    plt.show()
    plt.clf()

def prepare_plots(actions, actions_per_label=20):
    a1 = actions[:actions_per_label]
    a2 = actions[actions_per_label:2*actions_per_label]
    a3 = actions[actions_per_label*2:]
    js = ["Left Shoulder Roll", "Left Elbow Roll", "Right Shoulder Roll", "Right Elbow Roll"]
    means0, means1, means2, means3 = [], [], [], []
    for ind, m in enumerate([means0, means1, means2, means3]):
        for a in [a1, a2, a3]:
            m.append([mean(x, ind) for x in a])
        plot_hist(m[0], m[1], m[2], "Mean of " + js[ind], "mean{}".format(ind))
    sds0, sds1, sds2, sds3 = [], [], [], []
    for ind, m in enumerate([sds0, sds1, sds2, sds3]):
        for a in [a1, a2, a3]:
            m.append([stdev(x, ind) for x in a])
        plot_hist(m[0], m[1], m[2], "SD of " + js[ind], "sd{}".format(ind))

def convert_to_radians_all(what):
    new_d = []
    for i, x in enumerate(what):
        new_m = [math.radians(l) for l in x[:8]] + list(x[:] / 180)
        new_d.append(new_m)
    return new_d

def retrieve_samples(trained_models, samples_num):
    samples = []
    s_labels = []
    for k in trained_models.keys():
        samples_all = trained_models[k].generate_samples(samples_num*4, int(trained_models[k].classes[k]))
        labels_all = [k] * len(samples_all)
        actions_ok = check_action_ok(np.asarray([np.asarray(x.detach().cpu()) for x in samples_all]).squeeze(), labels_all)
        print("{} samples that are ok: {}/{}".format(k, sum(actions_ok), len(actions_ok)))
        selected_labels_idx = [i for i, p in enumerate(actions_ok) if p == 1][:samples_num]
        if len(selected_labels_idx) >= (int(samples_num/2)):
            samples_selected = [samples_all[i] for i in selected_labels_idx]
            labels_selected = [labels_all[i] for i in selected_labels_idx]
            samples += samples_selected
            s_labels += labels_selected
        else:
            samples += samples_all[:samples_num]
            s_labels += labels_all[:samples_num]
    return samples, s_labels

def data_replay(vae, trained_models, replay_samples, epochs=20):
    for _ in range(epochs):
        samples, s_labels = retrieve_samples(trained_models, replay_samples)
        sampled_data, sampled_labels = shuffle_twolists(samples, s_labels)
        print("Additional training on: ", sampled_labels)
        sampled_data = torch.tensor(torch.nn.utils.rnn.pad_sequence(sampled_data, batch_first=True, padding_value=0.0))
        vae.train_step(sampled_data, label=sampled_labels, replay=True)

def train_vae_offline(vae):
    ordered = True if int(vae.config["ordered"]) == 1 else False
    epochs = int(vae.config["epochs"])
    replay = True if int(vae.config["replay"]) == 1 else False
    replay_samples = int(vae.config["replay_samples"])
    datapath = vae.config["modality_1"]["path"]
    data, labels = load_actions_from_path(datapath, sort=ordered)
    batch_size = vae.batch_size
    idx = 0
    current_class, samples = None, None
    trained_models = {}
    while not idx == (int(len(data)/batch_size)):
         samples, s_labels = None, None
         for epoch in range(epochs):
             d = tuple(np.asarray(data)[idx * batch_size:(idx * batch_size + batch_size)])
             l = labels[idx*batch_size:(idx*batch_size+batch_size)]
             if current_class and current_class != l[0] and replay:
                 vae.save_model(tag="model_{}".format(current_class))
                 trained_models[current_class] = load_model(vae.runPath + '/model_{}.rar'.format(current_class))
                 if len(trained_models.keys())>1:
                     data_replay(vae, trained_models, replay_samples)
             #if len(trained_models.keys())> 0 and epoch == (epochs-1):
             #    samples, s_labels = retrieve_samples(trained_models, replay_samples)
             current_class = l[0]
             if batch_size > 1 or replay:
                 if not ordered:
                     d, l  = shuffle_twolists(d, l)
                 d = [torch.from_numpy(np.asarray(vae.pick_single(x))) for x in d]
                 if samples is not None:
                      for i,s in enumerate(samples):
                         d += [s]
                         l += [s_labels[i]]
                      d, l = shuffle_twolists(d, l)
                      print(l)
                 d = torch.tensor(torch.nn.utils.rnn.pad_sequence(d, batch_first=True, padding_value=0.0))
             else:
                d = np.expand_dims(np.asarray(vae.pick_single(d[0])), 0)
             vae.train_step(d, label=l)
         idx += 1
    if replay:
         vae.save_model(tag="model_{}".format(current_class))
         trained_models[current_class] = load_model(vae.runPath + '/model_{}.rar'.format(current_class))
         data_replay(vae, trained_models, replay_samples, epochs=100)
         vae.eval()
         vae.eval_progress()
    print("{} done at iter {}".format(os.path.basename(datapath), vae.iternum))
    plot_loss_action(vae.runPath, vae)
    plot_metrics(vae.runPath, vae=vae)


def shuffle_twolists(data, labels):
    temp = list(zip(data, labels))
    random.shuffle(temp)
    data, labels = zip(*temp)
    return data, labels

def vae_reconstruct(vaepath):
    model = load_model(vaepath)
    testdata, labels = load_actions_from_path("./dataset/testdata_20")
    t_check = (check_action_ok(np.asarray(testdata)/180, list(labels)))
    labels_t = [model.classes[x] for x in labels]
    testdata_c = np.asarray([np.asarray(pick_single(x, size=4)) for x in testdata])
    output = model.reconstruct(testdata_c, labels_t, N=100)
    o = np.asarray(output[1].loc.squeeze().detach().cpu())
    o_check = (check_action_ok(o, ["wave"] * len(testdata))[0])
    print("Testdata check: {} Output check: {}".format(t_check, o_check))
    return o, labels, testdata
