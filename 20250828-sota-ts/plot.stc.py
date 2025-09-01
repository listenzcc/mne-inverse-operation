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

data_directory = Path('./data/fsaverage/sub-1-ts')
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
