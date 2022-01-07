import os
import random
import numpy as np
import pandas as pd
from glob import glob

def extract_day(images):
    day = int(images.split('.')[-2][-2:])
    return day

def make_day(images):
    day_array = np.array([extract_day(x) for x in images])
    return day_array

def make_df(dir, images, days, type=''):
    dataframe = []
    for i in dir:
        df = pd.DataFrame({
            'file_name': images[i],
            'day': days[i],
            'species': type,
            'version': i
        })
        dataframe.append(df)
    type_df = pd.concat(dataframe).reset_index(drop=True)
    return type_df

def make_comb_df(species, data_frame):
    before_file_path = []
    after_file_path = []
    time_delta = []

    for version in data_frame[data_frame['species'] == species]['version'].unique():
        cnt = 0
        for i in range(0, len(data_frame[data_frame['version'] == version]) - 1):
            for j in range(i + 1, len(data_frame[data_frame['version'] == version])):
                before = data_frame[data_frame['version'] == version].iloc[i].reset_index(drop=True)
                after = data_frame[data_frame['version'] == version].iloc[j].reset_index(drop=True)
                delta = int(after[1] - before[1])

                if delta>0:
                    before_file_path.append(before[0])
                    after_file_path.append(after[0])
                    time_delta.append(delta)
                    cnt += 1

    combination_df = pd.DataFrame({
        'before_file_path': before_file_path,
        'after_file_path': after_file_path,
        'time_delta': time_delta,
    })
    combination_df['species'] = species
    return combination_df

if __name__ == '__main__':
    root_path = 'train_dataset'
        
    bc_path = glob(os.path.join(root_path, 'BC/*'))
    bc_dir = [x[-5:] for x in bc_path]

    lt_path = glob(os.path.join(root_path, 'LT/*'))
    lt_dir = [x[-5:] for x in lt_path]

    bc_images = {key: glob(name + '/*.png') for key, name in zip(bc_dir, bc_path)}
    lt_images = {key: glob(name + '/*.png') for key, name in zip(lt_dir, lt_path)}

    bc_days = {key: make_day(bc_images[key]) for key in bc_dir}
    lt_days = {key: make_day(lt_images[key]) for key in lt_dir}

    bc_df = make_df(bc_dir, bc_images, bc_days, type='bc')
    lt_df = make_df(lt_dir, lt_images, lt_days, type='lt')

    bc_lt_df = pd.concat([bc_df, lt_df]).reset_index(drop=True)

    bc_comb_df = make_comb_df('bc', bc_lt_df)
    lt_comb_df = make_comb_df('lt', bc_lt_df)
    bc_comb_df = bc_comb_df.sample(frac=1, random_state=34).reset_index(drop=True)
    lt_comb_df = lt_comb_df.sample(frac=1, random_state=34).reset_index(drop=True)

    bc_comb_df.to_pickle('bc_comb_df')
    lt_comb_df.to_pickle('lt_comb_df')