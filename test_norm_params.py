import os
import glob
import numpy as np

os.chdir("/home/ubuntu/DeepSDF/data/NormalizationParameters/NOCSTEST/02942699")

dirname = "/home/ubuntu/DeepSDF/data/NormalizationParameters/NOCSTEST/02942699"
inst_list = []
for file in glob.glob("*.npz"):
    inst_list.append(os.path.join(dirname,file ))

for norm_file in inst_list:
    print("norm_file", norm_file)
    normalization_params = np.load(
        norm_file
    )
    offset = normalization_params["offset"]
    scale = normalization_params["scale"]

    print("offset, scale", offset, scale)

    print("================\n")