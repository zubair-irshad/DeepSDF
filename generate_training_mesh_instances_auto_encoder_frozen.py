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
import _pickle as cPickle

def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

    specs_filename = os.path.join(experiment_directory, "specs.json")
    print("specs_filename", specs_filename)
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder", "PointCloudEncoder"])
    arch_encoder = __import__("networks." + "auto_encoder", fromlist=["PointCloudAE"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    encoder = arch.PointCloudEncoder(latent_size)

    decoder = torch.nn.DataParallel(decoder)
    encoder = torch.nn.DataParallel(encoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["decoder_model_state_dict"])

    model_path = '/home/ubuntu/object_deformnet/results/ae_nocs_train_sdf/model_50.pth'
    n_pts = 1024
    estimator = arch_encoder.PointCloudAE(latent_size, n_pts)
    estimator.cuda()
    estimator.load_state_dict(torch.load(model_path))
    estimator.eval()

    # encoder.load_state_dict(saved_model_state["encoder_model_state_dict"])

    decoder = decoder.module.cuda()
    # encoder = encoder.module.cuda()

    decoder.eval()
    # encoder.eval()

    data_source = specs["DataSource"]

    instances_names = ['bottle3_scene5_norm', 'bottle_blue_google_norm','bottle_starbuck_norm', 'bowl_blue_ikea_norm', 'bowl_brown_ikea_norm', 'bowl_chinese_blue_norm','camera_anastasia_norm','camera_dslr_len_norm', 'camera_dslr_wo_len_norm', 'can_milk_wangwang_norm', 'can_porridge_norm', 'can_tall_yellow_norm','laptop_air_0_norm', 'laptop_air_1_norm', 'laptop_dell_norm', 'mug2_scene3_norm', 'mug_vignesh_norm', 'mug_white_green_norm']
    obj_model_dir = os.path.join(data_source, ws.sdf_samples_subdir, 'NOCS_TRAIN_WATERTIGHT', 'camera_real_train_sdf.pkl')
    with open(obj_model_dir, 'rb') as f:
        obj_models = cPickle.load(f)


    latent_vectors = []

    for instance in instances_names:
        points = obj_models[instance]
        print(torch.from_numpy(points).shape)
        points = torch.from_numpy(points).unsqueeze(0).to(device = device, dtype=torch.float)
        latent_emb, _ = estimator(points)
        # latent_emb = encoder(points)
        latent_vectors.append(latent_emb)

    # latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)


    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)
    exp_dir = Path(experiment_directory).parts[0]
    print("exp_dir", exp_dir)
    category_name = 'all'
    instance_filename = os.path.join(
        exp_dir, 'splits', category_name+'_train_ids_modified.json'
    )
    instance_filename = Path(instance_filename)
    with open(instance_filename, "r") as f:
        instance_ids = json.load(f)

    indices = [instance_ids[k] for k,v in instance_ids.items() if k in instances_names]

    # new_latent_vecs = []
    new_instance_filenames = []
    for l in range(len(indices)):
        # new_latent_vecs.append(latent_vectors[indices[l]])
        new_instance_filenames.append(instance_filenames[indices[l]])
    # latent_vectors = new_latent_vecs 
    instance_filenames = new_instance_filenames

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
