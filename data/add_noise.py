# to use only portion of data to train
import os
import pathlib
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

mu, sigma = 0, 0.1
strength = 9
csv_files = []

root = 'A_DeviceMotion_data'
target_root = 'A_DeviceMotion_data_{}_{}_{}'.format(mu, sigma, strength)


def get_new_path(current_path, anchor, new_head):
    path = pathlib.Path(current_path)
    index = path.parts.index(anchor)
    new_path = pathlib.Path(new_head).joinpath(*path.parts[index + 1:])
    new_dir = pathlib.Path('.').joinpath(*new_path.parts[:-1])
    return new_path, new_dir


for path, subdirs, files in os.walk(root):
    for name in files:
        csv_files.append(os.path.join(path, name))


def create_path_and_save(df, new_file, new_path):
    new_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(new_file), index=False)


def add_noise(df, file_name):
    indexes = [m.start() for m in re.finditer('/', file_name)]
    move = file_name[indexes[0] + 1: indexes[1]]
    number = int(move[move.index('_') + 1:])
    if number > 10:
        return df
    noise = np.random.normal(mu, sigma, [df.shape[0], df.shape[1]])
    return df + noise * strength


for f in tqdm(csv_files):
    file = open(f)
    total_len = len(file.readlines())
    df = pd.read_csv(f)
    df_with_noise = add_noise(df, f)
    new_file, new_path = get_new_path(f, root, target_root)
    create_path_and_save(df_with_noise, new_file, new_path)

# create zip root
target_zip_root = target_root + '_zip'
output_dir = pathlib.Path(target_zip_root)
output_dir.mkdir(parents=True, exist_ok=True)


# create zip
def create_zip(dest_zip, source_folder):
    import os, zipfile
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(source_folder):
            # folder_name = folder_name.replace('data_{}'.format(int(percentage * 100)), 'data')
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, source_folder))

    zip_ref.close()


create_zip(
    dest_zip=target_zip_root + '/' + 'A_DeviceMotion_data.zip',
    source_folder=target_root
)
