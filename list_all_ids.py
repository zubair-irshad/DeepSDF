
import json
import os
train_split_file = '/home/ubuntu/DeepSDF/examples/splits/all_train_modified.json'
with open(train_split_file, "r") as f:
    train_split = json.load(f)

import deep_sdf.workspace as ws
data_source = "data"
split = train_split



def get_instance_filenames(data_source, split):
    count = 0
    npzfiles = []
    inst_name = []
    inst_dict = {}
    seen_names = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if instance_name in seen_names:
                    print("instance name", instance_name)
                inst_dict[instance_name] = count
                inst_name.append(instance_name)
                count+=1
                npzfiles += [instance_filename]
                seen_names.append(instance_name)
    return inst_dict, npzfiles, inst_name

inst_dict, npzfiles, inst_name = get_instance_filenames(data_source, split)

json_save_folder = '/home/ubuntu/DeepSDF/examples/splits'
filename_ids = 'all_train_ids_modified.json'
json_filename_ids = os.path.join(json_save_folder,filename_ids)
with open(json_filename_ids, 'w') as fp:
    json.dump(inst_dict, fp)


# print(npzfiles)
# print(inst_dict)
print("inst_dict", len(inst_dict.keys()))
print("print(len(npzfiles))", len(npzfiles))

print("inst_name", len(inst_name))

a = []
for k,v in inst_dict.items():
    a.append(k)

print(len(a))
