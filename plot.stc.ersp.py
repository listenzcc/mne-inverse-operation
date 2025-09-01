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

data_directory = Path('./data/fsaverage/sub-1-ersp')
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
clim = {
    'kind': 'value',
    'lims': [0.0, 0.3, 0.6]
}

# %%
brain_kwargs = dict(alpha=0.8, background="white", cortex="low_contrast")
output_directory = Path('./img/ersp')
output_directory.mkdir(exist_ok=True, parents=True)

for evt in tqdm(['T80', 'T100', 'T120', 'Sham']):
    stc = get_stc(evt, fix_scale=True)
    for t in tqdm(np.linspace(0, 0.21, 5, endpoint=False)):
        brain = stc.plot(
            initial_time=t,
            hemi="both",
            views=['dorsal'],
            surface='inflated',
            subjects_dir=SubjectFsaverage.subjects_dir,
            transparent=True,
            show_traces=False,
            clim=clim,
            brain_kwargs=brain_kwargs
        )
        brain.add_text(0.1, 0.9, f'{evt}-{t:0.2f}', 'title', font_size=16)

        # 1. 截图
        screenshot = brain.screenshot()

        # 2. 保存图像
        brain.save_image(output_directory / f'{evt=}-{t=:0.2f}.png')
        brain.close()


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

    # Plot with stc.plot
    # <https://mne.tools/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate.plot>
    brain_kwargs = dict(alpha=0.8, background="white", cortex="low_contrast")
    brain = stc.plot(
        initial_time=0,
        # hemi="split",
        hemi="both",
        views=['dorsal'],
        # surface='pial',
        surface='inflated',
        subjects_dir=SubjectFsaverage.subjects_dir,
        transparent=True,
        # show_traces=False,
        clim=clim,
        brain_kwargs=brain_kwargs
    )

    # for hemi in ['lh', 'rh']:
    #     brain.add_label("BA4a", hemi=hemi, color="green", borders=True)
    #     brain.add_label("BA4p", hemi=hemi, color="blue", borders=True)
    brain.add_text(0.1, 0.9, evt, 'title', font_size=16)

    print(dir(brain))
    # keys = list(brain._picked_points.keys())
    # for key in keys:
    #     print(key)
    #     brain._picked_points.pop(key)


sys.exit(0)

# %%
