# %% ---- 2025-06-27 ------------------------
from util.easy_import import *

compile = re.compile(r'^(?P<mode>[a-z]+)-evt(?P<evt>[a-zA-z0-9]+).stc-lh.stc')


# %%
class SubjectFsaverage:
    # MNE fsaverage
    subject = 'fsaverage'
    subject_dir = mne.datasets.fetch_fsaverage()
    subjects_dir = subject_dir.parent


def find_stc_files(pattern: str, data_directory: Path):
    found = list(data_directory.rglob(pattern))
    return found


def mk_stc_file_table(stc_files: list):
    data = []
    for p in stc_files:
        name = p.name
        dct = compile.search(name).groupdict()
        dct.update({
            'path': p,
            'virtualPath': p.parent.joinpath(p.name.replace('.stc-lh', ''))
        })
        data.append(dct)
    df = pd.DataFrame(data)
    df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
    return df


# %%
subject = SubjectFsaverage()

data_directory = Path('./data/fsaverage')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
table = mk_stc_file_table(stc_files)
print(stc_files)
print(table)

# %%


def get_stc(evt, fix_scale=False):
    stc = table.query(f'mode=="eeg" & evt=="{evt}"').iloc[0]['stc'].copy()
    stc.subject = subject.subject

    if fix_scale:
        s = get_stc('T120')
        max_s = np.max(s.data)
        stc.data /= max_s

    return stc


# %%
while True:
    evt = input('Input like T80 | T100 | T120 | Sham | q >> ')
    evt = evt.strip().title()
    if evt.lower() == 'q':
        break
    elif evt == '':
        continue

    print(evt)

    stc = get_stc(evt, fix_scale=True)
    print(stc)

    brain = stc.plot(
        initial_time=1,
        # hemi="split",
        hemi="both",
        views=['dorsal'],
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
        title=evt,
    )
    brain.add_text(0.1, 0.9, evt, 'title', font_size=16)

sys.exit(0)

# %%

# %%

df1 = mk_stc_file_table(stc_files)
df1['band'] = 'alpha'
data_directory = Path('./data/tfr-stc-beta')
stc_files = find_stc_files('*.stc-lh.stc', data_directory)
df2 = mk_stc_file_table(stc_files)
df2['band'] = 'beta'
df = pd.concat([df1, df2])

print('Reading stc')
df['stc'] = df['virtualPath'].map(mne.read_source_estimate)
print(df)

# %% ---- 2025-06-27 ------------------------
# Play ground


def plot_brain(mode, evt, band):
    conditions = [
        f'me=="{mode}"',
        f'evt=="{evt}"',
        f'band=="{band}"',
    ]

    selected = df.query('&'.join(conditions))
    print(selected)

    # Average stcs
    stc = selected.iloc[0]['stc']
    mat = np.zeros_like(stc.data)
    for s in selected['stc'].values:
        mat += s.data
    mat /= len(selected)
    stc.data = mat

    # Prepare the stc
    stc.subject = subject.subject
    stc.crop(tmin=-2, tmax=4.5)
    stc.apply_baseline((-1, 0))
    print(stc)

    # Plot in 3D view
    brain = stc.plot(
        initial_time=1,
        hemi="split",
        views=["lat", "med"],
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
    )
    return brain


def plot_brain_sub_evts(mode, evt1, evt2, band):
    # evt1
    conditions1 = [
        f'me=="{mode}"',
        f'evt=="{evt1}"',
        f'band=="{band}"',
    ]
    selected1 = df.query('&'.join(conditions1))

    # evt2
    conditions2 = [
        f'me=="{mode}"',
        f'evt=="{evt2}"',
        f'band=="{band}"',
    ]
    selected2 = df.query('&'.join(conditions2))

    stc = selected1.iloc[0]['stc']

    # Average stcs
    mat1 = np.zeros_like(stc.data)
    for s in selected1['stc'].values:
        mat1 += s.data
    mat1 /= len(selected1)

    mat2 = np.zeros_like(stc.data)
    for s in selected2['stc'].values:
        mat2 += s.data
    mat2 /= len(selected2)

    stc.data = mat1 - mat2

    # Prepare the stc
    stc.subject = subject.subject
    stc.crop(tmin=-2, tmax=4.5)
    # stc.apply_baseline((-1, 0))

    # Plot in 3D view
    brain = stc.plot(
        initial_time=1,
        hemi="split",
        views=["lat", "med"],
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
        clim=dict(kind="value", pos_lims=(50, 75, 100)),
    )
    return brain


# %%
# ! CLI
while True:
    while True:
        inp = input('Input like meg-1-alpha | sub-meg-1-2-alpha | q >> ')
        if inp == 'q':
            break

        if inp.startswith('sub'):
            try:
                _, mode, evt1, evt2, band = inp.split('-')
                plot_brain_sub_evts(mode, evt1, evt2, band)
            except:
                pass

        try:
            mode, evt, band = inp.split('-')
            plot_brain(mode, evt, band)
        except:
            pass

    print('ByeBye')
    import sys
    sys.exit(0)


# %%
