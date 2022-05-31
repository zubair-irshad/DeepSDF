#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 6))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # print("colors",colors)
    label=['bowl', 'can', 'camera', 'mug', 'laptop', 'bottle']

    labels_dict={0:'bowl', 1:'can', 2:'camera', 3:'mug', 4:'laptop', 5:'bottle'}
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
    #                 c=palette[colors.astype(np.int)])
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=colors)

    # legend1 = ax.legend(*sc.legend_elements(),[labels_dict[i] for i in range(6)],
    #                     loc="upper left", title="Ranking")
    # ax.add_artist(legend1)
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
    #                 c=palette[colors])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')
    ax.set_xlabel('t-SNE-1')
    ax.set_ylabel('t-SNE-2')

    # labels = []
    # for i in range(6):
    #     color = colors[colors ==i]
    #     print(color)
    #     # print(palette[color.astype(np.int)]))
    #     print(palette[color.astype(np.int)][0])
    #     labels.append(mpatches.Patch(color=palette[color.astype(np.int)][0], label=label[i]))
    # lg = ax.legend(handles=labels,  bbox_to_anchor = (1.10 , 0.6), fontsize='small')
    # txts = []
    # for i in range(6):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    txts = None
    lg = None

    return f, ax, sc, txts,lg

def load_colors_latent():
    rgbnet_dict = torch.load(os.path.join('/home/ubuntu/Downloads', 'feats_colors.pt'))
    colors = rgbnet_dict['colors']
    print("len colors", len(colors))
    return colors


def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

    # category_num = {"bottle": 258, "bowl":147, "camera":77, "can":52, "laptop":394, "mug":174}

    category_num = {"bowl":147,"can":52, "camera":77,"mug":174, "laptop":394, "bottle": 258}
    colors = []
    for i, (k,v) in enumerate(category_num.items()):
        colors.append(i*np.ones(v)) 

    colors = np.concatenate(colors, axis=0)
    print("colors shape",colors.shape)
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    colors = load_colors_latent()

    instance_filename = os.path.join(
        '/home/ubuntu/DeepSDF/examples', 'splits', 'all_train_ids_modified.json'
    )
    print("instance filenameeeeeeeeeeeeeeeeeeeeee", instance_filename)
    instance_filename = Path(instance_filename)
    with open(instance_filename, "r") as f:
        instance_ids = json.load(f)
    
    colors_texture = []
    for k,v in instance_ids.items():
        if k not in colors:
            color = np.array([0.2668, 0.2637, 0.2659])
            # colors_texture.append(np.expand_dims(color, axis=0))
            colors_texture.append(color)
        else:
            # colors_texture.append(np.expand_dims(colors[k], axis=0))
            colors_texture.append(np.clip(colors[k].numpy(), a_min=0, a_max=1))

    print("colors_texture", len(colors_texture), colors_texture[0].shape)
    # colors = np.concatenate(colors_texture, axis=0)
    colors = colors_texture
    # print("colors", colors.shape)

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    RS = 20150101

    tsne = TSNE(random_state=RS).fit_transform(latent_vectors.numpy())
    f, ax, sc, txts,lg = scatter(tsne, colors)
    # plt.savefig('images/sup+contrastive_no_reg_textured.png', dpi=120,bbox_extra_artists=(lg,), 
    #         bbox_inches='tight')
    plt.savefig('images/sup+contrastive_no_reg_textured.png', dpi=120, 
            bbox_inches='tight')

    # print(len(instance_filenames), " vs ", len(latent_vectors))
    # print("latent_vectors", latent_vectors.shape)
    mean = torch.mean(latent_vectors, axis = 1)
    std = torch.std(latent_vectors, axis = 1)

    # print("mean", mean.numpy())
    # print("std", std.numpy())

    # print("mean", torch.mean(mean))
    # print("std", torch.mean(std))

    start = 0

    for k,v in category_num.items():
        latent_vecs_category = latent_vectors[start: start+v-1]
        # print("mean of category",k, torch.mean(torch.mean(latent_vecs_category, axis = 1)))
        # print("std of category",k, torch.mean(torch.std(latent_vecs_category, axis = 1)))
        # print("==================\n")


    # for i, latent_vector in enumerate(tqdm(latent_vectors)):
    #     print("mean latent vector", torch.mean(latent_vector))
    #     print("std latent vec", )
        # print("latent vector", latent_vector)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        dest="keep_normalized",
        default=False,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint, args.keep_normalized)
