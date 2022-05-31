#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws
from pathlib import Path

def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

    snc_synth_id_to_category_nocs_camera = {
        '02876657': 'bottle',    '02880940': 'bowl',       '02942699': 'camera',        
        '02946921': 'can',    '03642806': 'laptop',      '03797390': 'mug',
    }

    snc_synth_category_to_id_nocs_camera = {
        'bottle':'02876657',    'bowl':'02880940' ,       'camera':'02942699',        
        'can':'02946921',    'laptop':'03642806',      'mug':'03797390',
    }

    specs_filename = os.path.join(experiment_directory, "specs.json")
    print("specs_filename", specs_filename)
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

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    print(len(instance_filenames), " vs ", len(latent_vectors))

    print("instance filenames", instance_filenames)

    #instances_names = ['bottle3_scene5_norm', 'bottle_blue_google_norm','bottle_starbuck_norm', 'bowl_blue_ikea_norm', 'bowl_brown_ikea_norm', 'bowl_chinese_blue_norm','camera_anastasia_norm','camera_dslr_len_norm', 'camera_dslr_wo_len_norm', 'can_milk_wangwang_norm', 'can_porridge_norm', 'can_tall_yellow_norm','laptop_air_0_norm', 'laptop_air_1_norm', 'laptop_dell_norm', 'mug2_scene3_norm', 'mug_vignesh_norm', 'mug_white_green_norm']
    instances_names = ['camera_anastasia_norm','camera_dslr_len_norm', 'camera_dslr_wo_len_norm']
    print(experiment_directory)
    print("len instance names", len(instances_names))
    exp_dir = Path(experiment_directory).parts[0]
    category_name = Path(experiment_directory).parts[1]
    category_name = 'all'
    print("category name", category_name)
    instance_filename = os.path.join(
        exp_dir, 'splits', category_name+'_train_ids_modified.json'
    )
    print("instance filenameeeeeeeeeeeeeeeeeeeeee", instance_filename)
    instance_filename = Path(instance_filename)
    with open(instance_filename, "r") as f:
        instance_ids = json.load(f)

    print("len", len(instance_filenames), len(latent_vectors))
    print("instance ids", instance_ids)

    indices = [instance_ids[k] for k,v in instance_ids.items() if k in instances_names]

    print("indices, len", indices, len(indices))
    indices = np.array(indices).astype(int)
    # latent_vectors = np.array(latent_vectors)[indices].tolist()
    new_latent_vecs = []
    new_instance_filenames = []
    for l in range(len(indices)):
        new_latent_vecs.append(latent_vectors[indices[l]])
        new_instance_filenames.append(instance_filenames[indices[l]])
    latent_vectors = new_latent_vecs 
    instance_filenames = new_instance_filenames
    print("instance_filenames", instance_filenames)
    # instance_filenames = np.array(instance_filenames)[indices].tolist()

    for i, latent_vector in enumerate(latent_vectors):

        dataset_name, class_name, instance_name = instance_filenames[i].split("/")
        instance_name = instance_name.split(".")[0]

        print("{} {} {}".format(dataset_name, class_name, instance_name))
        # dataset_name = 'Real_Meshes'
        dataset_name_saving = 'Real_Meshes'
        # mesh_dir = os.path.join(
        #     experiment_directory,
        #     ws.training_meshes_subdir,
        #     str(saved_model_epoch),
        #     dataset_name,
        #     class_name,
        # )
        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name_saving,
            class_name,
        )
        print(mesh_dir)

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        print(instance_filenames[i])

        offset = None
        scale = None

        if not keep_normalized:

            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        # print("latent vector", latent_vector.device, latent_vector.shape)
        latent_vector = latent_vector.cuda()
        with torch.no_grad():
            deep_sdf.mesh.create_mesh(
                decoder,
                latent_vector,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18),
                offset=offset,
                scale=scale,
            )


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
        default="2000",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        dest="keep_normalized",
        default=True,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint, args.keep_normalized)
