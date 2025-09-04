"""
File: draw_stc_in_label.py
Author: Chuncheng Zhang
Date: 2025-09-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Draw the timeseries of the stc as in_label.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-01 ------------------------
# Requirements and constants
from util.easy_import import *

# %%
data_directory = Path('./data')

try:
    label_data = pd.read_hdf(
        data_directory.joinpath('label_data.h5'), key='h5')
except:
    label_path = Path('./label')
    label_files = list(label_path.glob('*.label'))
    annot_files = list(label_path.glob('*.annot'))
    label_data = []
    for file in label_files:
        label = mne.read_label(file)
        label_data.append(['onlyLabel', label.name, label])
    for file in annot_files:
        labels = mne.read_labels_from_annot('fsaverage', annot_fname=file)
        for label in labels:
            if label.name.startswith('Unknown'):
                continue
            label_data.append([file.name, label.name, label])
    label_data = pd.DataFrame(label_data, columns=['annot', 'name', 'label'])
    label_data.to_hdf(data_directory.joinpath('label_data.h5'), key='h5')

print(label_data)

# %%

# %% ---- 2025-09-01 ------------------------
# Function and class
if __name__ == '__main__':
    labels = label_data.query('annot == "onlyLabel"')['label']
    for mode in tqdm(['ersp', 'ts']):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for evt in tqdm(['T120', 'T100', 'T80', 'Sham']):
            p = data_directory.joinpath(
                f'fsaverage/sub-1-{mode}/eeg-evt{evt}.stc')
            stc = mne.read_source_estimate(p)
            if evt == 'T120':
                m = np.max(stc.data)
            for label in labels:
                _stc = stc.in_label(label)

                d = np.mean(_stc.data, axis=0)/m
                df['times'] = _stc.times
                df['mode'] = mode
                df[f'{evt}_{label.name}'] = d

                d = np.std(_stc.data, axis=0)/m
                df2['times'] = _stc.times
                df2['mode'] = mode
                df2[f'{evt}_{label.name}'] = d
        print(df)
        df.to_csv(data_directory.joinpath(f'{mode}.average.csv'))
        df2.to_csv(data_directory.joinpath(f'{mode}.std.csv'))

    print(stc)

    label = label_data.iloc[30]['label']
    _stc = stc.in_label(label)
    print(_stc)
    import plotly.express as px
    df = pd.DataFrame()
    df['times'] = _stc.times
    df['value'] = np.mean(_stc.data, axis=0)
    df['name'] = label.name
    fig = px.scatter(df, x='times', y='value', color='name')
    fig.show()


# %% ---- 2025-09-01 ------------------------
# Play ground


# %% ---- 2025-09-01 ------------------------
# Pending


# %% ---- 2025-09-01 ------------------------
# Pending
