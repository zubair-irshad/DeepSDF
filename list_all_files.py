import numpy
import os
import json

DATAFILES_FOLDER = '/home/ubuntu/rgbd/fleet/simnet/nocs_data/obj_models_watertight/val'
json_save_folder = '/home/ubuntu/DeepSDF/examples/splits'
folders = os.listdir(DATAFILES_FOLDER)
snc_synth_id_to_category_nocs_camera = {
    '02876657': 'bottle',    '02880940': 'bowl',       '02942699': 'camera',        
    '02946921': 'can',    '03642806': 'laptop',      '03797390': 'mug',
}
category_dict = {}
category_dict['NOCS_VAL_WATERTIGHT_WS'] = {}
count = 0
inst_dict = {}
for catId, class_folder in enumerate(folders):

    # category_dict = {}
    # category_dict['NOCS_VAL_WATERTIGHT_WS'] = {}
    if class_folder not in snc_synth_id_to_category_nocs_camera.keys():
        continue
    category = snc_synth_id_to_category_nocs_camera[class_folder]
    
    print("categoey", category)
    filename = category+'_train.json'
    json_filename = os.path.join(json_save_folder,filename)
    synset_dir = os.path.join(DATAFILES_FOLDER, class_folder)
    inst_list = sorted(os.listdir(synset_dir))

    # for i, inst in enumerate(inst_list):
    #     inst_dict[inst] = count+i

    category_dict['NOCS_VAL_WATERTIGHT_WS'][class_folder] = inst_list
    # with open(json_filename, 'w') as fp:
    #     json.dump(category_dict, fp)
    # print("inst_list", inst_list)
    # print("=========================\n\n")
    # count+=i+1

# filename_ids = 'all_train_ids.json'
# json_filename_ids = os.path.join(json_save_folder,filename_ids)
# with open(json_filename_ids, 'w') as fp:
#     json.dump(inst_dict, fp)

json_filename = os.path.join(json_save_folder,'all_val_modified_ws.json')
with open(json_filename, 'w') as fp:
    json.dump(category_dict, fp)



