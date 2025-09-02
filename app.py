# %%
from dash import Dash, html, dcc, callback, Output, Input
from draw_stc_in_label import label_data, mne, data_directory, np
import plotly.express as px
import pandas as pd

# %%

# df = pd.read_csv(
#     'https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children='STC data drawer', style={'textAlign': 'center'}),
    # dcc.Dropdown(['T120', 'T100', 'T80', 'Sham'], 'T120', id='evt-selection'),
    dcc.Dropdown(label_data['annot'].unique(), label_data.iloc[0]
                 ['annot'], id='annot-selection'),
    dcc.Dropdown(label_data['name'], label_data.iloc[0]
                 ['name'], id='name-selection'),
    dcc.Dropdown([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                 0.9, 1.0], 0.3, id='threshold-selection'),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
]


@callback(
    Output('name-selection', 'options'),
    Input('annot-selection', 'value'),
)
def update_annot_selection(annot):
    df = label_data.query(f'annot == "{annot}"')

    options = [{'label': name, 'value': f'annot=="{annot}" & name=="{name}"'}
               for name, label
               in zip(df['name'], df['label'])]

    return options


@callback(
    Output('graph-content', 'figure'),
    [Input('name-selection', 'value'),
        Input('threshold-selection', 'value')]
)
def update_name_selection(query, threshold):
    print(query, threshold)
    if query is None:
        return None

    if threshold is None:
        return None

    label = label_data.query(query).iloc[0]['label']

    dfs = []
    stcs = {}
    for evt in ['T120', 'T100', 'T80', 'Sham']:
        stcs[evt] = mne.read_source_estimate(data_directory.joinpath(
            f'fsaverage/sub-1-ersp/eeg-evt{evt}.stc'))

    m = np.max(stcs['T120'].data)

    for evt, stc in stcs.items():
        _stc = stc.in_label(label)

        df = pd.DataFrame()
        df['times'] = _stc.times

        d = []
        for _d in _stc.data:
            if np.max(_d) / m > float(threshold):
                d.append(_d)

        if not d:
            d = np.zeros_like(_stc.data)

        df['value'] = np.mean(d, axis=0)/m
        df['name'] = f'{evt=}, {label.name}'
        dfs.append(df)

    df = pd.concat(dfs)

    fig = px.scatter(df, x='times', y='value',
                     color='name', title=f'{threshold=}')
    return fig


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='59215')
