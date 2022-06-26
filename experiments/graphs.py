import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

def add_line(fg, **kwargs):
    fg.add_trace(
        go.Scatter(
            mode='lines',
            **kwargs
        ),
    )

def score_graph(batches, dfs, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Score X Generation {seeds}')
    )

    for batch in batches:
        df = dfs.get(batch)
        add_line(fg, x=df.gen, y=df.specialist_score, name=f'Score - Batch [{batch}]')

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Score')
    fg.show()

def precision_recall_graph(dfs, batches, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Precision/Recall X Generation {seeds}')
    )

    for batch in batches:
        df = dfs.get(batch)
        add_line(fg, x=df.gen, y=df.precision, name=f'Precision - Batch {batch}')
        add_line(fg, x=df.gen, y=df.recall, name=f'Recall - Batch {batch}')

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Precision/Recall')
    fg.show()

def heatmap(df, title='Title', colorscale='temps_r'):
    fg = go.Figure(
        layout=go.Layout(title=title)
    )
    fg.add_heatmap(z=df, colorscale=colorscale)

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Conditions')
    fg.show()
