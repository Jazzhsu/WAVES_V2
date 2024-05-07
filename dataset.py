from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
import json
from tqdm import tqdm
import os

# load data from HF
# TODO: REMOVE THIS FILE when release.
path = './.cache/OpenDatasets___dalle-3-dataset/default/0.0.0/65a6a9746a38e0aa516fe0a169e60dfce83c9cee/'

# dataset = load_dataset('OpenDatasets/dalle-3-dataset', cache_dir='./.cache', ignore_verifications=True)
dataset = concatenate_datasets([
    Dataset.from_file(os.path.join(path, file_name)) for file_name in os.listdir('./.cache/OpenDatasets___dalle-3-dataset/default/0.0.0/65a6a9746a38e0aa516fe0a169e60dfce83c9cee/') if file_name.endswith('arrow')
])

out_dir = './image_data/'
target_size = (1024, 1024)
target_data_size = 5000
data_count = 500000
caption_map = {}
bar = tqdm(total=target_data_size)
for i in tqdm(range(data_count)):
    file_name = f'{len(caption_map):05d}.png'
    im = dataset[i]['image']
    if im.size[0] == im.size[1]:
        im.resize((512, 512)).save(f'{out_dir}{file_name}')
        caption_map[file_name] = dataset[i]['caption']
        bar.update(1)

    if len(caption_map) == target_data_size:
        break

with open(f'{out_dir}caption.json', 'w', encoding='utf-8') as f:
    json.dump(caption_map, f, ensure_ascii=False, indent=4)

