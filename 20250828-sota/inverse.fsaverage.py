"""
File: inverse.fsaverage.py
Author: Chuncheng Zhang
Date: 2025-08-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read fsaverage files for fwd.

    By looking at Display sensitivity maps for EEG and MEG sensors plot the sensitivity maps for EEG and compare it with the MEG, can you justify the claims that:
    <https://mne.tools/stable/auto_examples/forward/forward_sensitivity_maps.html#ex-sensitivity-maps>
    <https://mne.tools/stable/generated/mne.sensitivity_map.html#mne.sensitivity_map>

    - MEG is not sensitive to radial sources
    - EEG is more sensitive to deep sources

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-27 ------------------------
# Requirements and constants
import joblib
import sys
import io
from contextlib import redirect_stdout

from util.easy_import import *
from util.subject_fsaverage import SubjectFsaverage
from read_raw import evokeds, ea

subject_name = 'sub-1'

data_directory = Path(f'./data/fsaverage/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-06-27 ------------------------
# Function and class
evoked_sham = evokeds['Sham']
info = evoked_sham.info


# %%

# Prepare subject
subject = SubjectFsaverage()
subject.pipeline()
fwd = subject.read_forward_solution(info, 'eeg')
print('src', subject.src)
print('bem', subject.bem)

# Compute cov
print('Computing noise_cov')
with redirect_stdout(io.StringIO()):
    # data = evoked_sham.data.copy()
    # times = evoked_sham.times
    # # data = data[:, times < 0]
    # cov = mne.Covariance(np.corrcoef(data), names=info['ch_names'], bads=[
    # ], projs=[], nfree=data.shape[1]-1)

    # # evoked_sham.set_eeg_reference('average', projection=True)
    # # cov = mne.compute_covariance([evoked_sham], method=['empirical'])

    # noise_cov = dict(eeg=cov)

    method = ['empirical', 'shrunk']
    noise_cov = dict(
        eeg=mne.compute_covariance(ea, tmin=-0.1, tmax=-0.05, method=method),
    )


print(noise_cov)

# Compute inverse operator
print('Computing inverse_operator')
with redirect_stdout(io.StringIO()):
    inverse_operator = dict(
        eeg=subject.make_inverse_operator(
            info, fwd, noise_cov['eeg']),
    )
print(inverse_operator)

stuff_estimate_snr = dict(
    cov_eeg=noise_cov['eeg'],
    fwd_eeg=fwd,
    info_eeg=info,
)

joblib.dump(stuff_estimate_snr, data_directory.joinpath(
    'stuff-estimate-snr.dump'))

# Compute inverse
snr = 3.0  # Standard assumption for average data but using it for single trial
kwargs = dict(
    lambda2=1.0 / snr**2,
    method="MNE",  # use dSPM method (could also be MNE or sLORETA)
)

evts = list(evokeds.keys())

# Compute EEG inverse
print('Computing EEG inverse')
with redirect_stdout(io.StringIO()):
    # stc = mne.minimum_norm.apply_inverse(ea, inverse_operator['eeg'], **kwargs)
    ea.set_eeg_reference(projection=True)

    for evt in evts:
        # evoked = evokeds[evt]
        evoked = ea[evt].average()
        projs = mne.compute_proj_evoked(evoked_sham)
        evoked.add_proj(projs)
        eeg_stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator['eeg'], **kwargs)
        print(eeg_stc)
        eeg_stc.save(data_directory.joinpath(
            f'eeg-evt{evt}.stc'), overwrite=True)

# sys.exit(0)

# %% ---- 2025-06-27 ------------------------
# Pending


# %% ---- 2025-06-27 ------------------------
# Pending
