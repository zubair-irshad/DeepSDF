import torch
import os

def load_latent_code(filename):
    data = torch.load(filename)
    return data.data.detach().squeeze(0).squeeze(0)

LATENT_CODE_DIR_NAME = '/home/ubuntu/DeepSDF/examples/all_ws_no_reg_contrastive0.1/Reconstructions/2000/Codes/NOCS_VAL_WATERTIGHT_WS'
    # category_name = snc_synth_id_to_category_nocs_camera[sysnet_list[i]]
category_name = 'all_ws_no_reg_contrastive0.1'
latent_filename = os.path.join(
LATENT_CODE_DIR_NAME, 'bottle_red_stanford_norm' + ".pth"
)
latent_vector = load_latent_code(latent_filename)

print(latent_vector.shape)
latent_embeddings.append(latent_vectors[index].cpu().numpy())