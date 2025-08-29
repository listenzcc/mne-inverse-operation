"""
File: read_raw.py
Author: Chuncheng Zhang
Date: 2025-08-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read the Sample data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-27 ------------------------
# Requirements and constants
from util.easy_import import *
from scipy import io as sio

path = Path('./rawdata/Sample/dongxiangyang_0308.vhdr')
sensors = open(Path('./rawdata/sensors.order.txt')).read().split()

# %%
data = {}
for key in ['T80', 'T100', 'T120', 'sham']:
    query_key = f'ERSP_{key}'
    loaded = sio.loadmat(f'./rawdata/ersp/ERSP_{key}.mat')
    data[key.title()] = loaded[query_key]
    times = loaded['times']
    print(f'{key.title()} shape:', data[key.title()].shape)

times

# %% ---- 2025-08-27 ------------------------
# Function and class
raw = mne.io.read_raw_brainvision(path)
raw.set_montage('standard_1020', on_missing='error')
raw.pick(sensors)
# raw.resample(1000)  # 5000 Hz -> 1000 Hz
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.15, tmax=0.35, baseline=(None, 0))

evoked = epochs.average()
evoked.resample(400)
evoked


# %%
epochs.ch_names

# %%
array = np.stack([data['T80'], data['T100'], data['T120'], data['Sham']])
events = np.zeros((array.shape[0], 3), int)
events[:, 0] = np.arange(array.shape[0])
events[:, 2] = np.arange(array.shape[0])
events[:, 2] += 1  # event_id start from 1
event_id = {'T80': 1, 'T100': 2, 'T120': 3, 'Sham': 4}
ea = mne.EpochsArray(array, info=evoked.info, tmin=-0.15,
                     event_id=event_id, events=events)
# ea.crop(tmin=-0.1, tmax=0.3)
ea.apply_baseline((None, 0))
ea

# %%

# %%

# %%

# %%
evoked = epochs.average()
evoked.resample(400)
print(f'{raw=}, {evoked=}, {events=}, {event_id=}')
print(f'{evoked.data.shape=}')

evokeds = {}
for key, value in data.items():
    evoked.data = value
    evoked.comment = key
    evoked.nave = 20
    evokeds[key] = evoked.copy()

evoked_sham = evokeds['Sham']
for key, evoked in evokeds.items():
    projs = mne.compute_proj_evoked(evoked_sham)
    evoked.add_proj(projs)
    evoked.set_eeg_reference(projection=True)

print(evokeds)


# %% ---- 2025-08-27 ------------------------
# Play ground
if __name__ == '__main__':
    print(raw.info)
    fig = raw.plot_sensors(show_names=True, to_sphere=False)
    plt.show()

    for key, evoked in evokeds.items():
        fig = evoked.plot(spatial_colors=True, titles=key)
        plt.show()

        evoked = ea[key].average()
        evoked.plot(spatial_colors=True, titles=f'{key=} (ea)')
        plt.show()


# %% ---- 2025-08-27 ------------------------
# Pending


# %% ---- 2025-08-27 ------------------------
# Pending

# %%
