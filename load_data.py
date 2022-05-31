import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import deep_sdf
from deep_sdf.data import unpack_sdf_samples
import deep_sdf.workspace as ws
train_split = "examples/splits/sv2_nocs_real_train.json"
data_source = "data"
num_samp_per_scene = 16384

# sdf_dataset = deep_sdf.data.SDFSamples(
#     data_source, train_split, num_samp_per_scene, load_ram=False
# )

# sdf_loader = data_utils.DataLoader(
#     sdf_dataset,
#     batch_size=scene_per_batch,
#     shuffle=True,
#     num_workers=num_data_loader_threads,
#     drop_last=True,
# )

filename = "/home/ubuntu/DeepSDF/data/SdfSamples/NOCSReal/000001/laptop.npz"

samples = unpack_sdf_samples(filename, 16384)

print(samples.shape)