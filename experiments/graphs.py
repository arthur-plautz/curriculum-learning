import plotly.graph_objects as go

def add_line(fg, **kwargs):
    fg.add_trace(
        go.Scatter(
            mode='lines',
            **kwargs
        ),
    )

def score_group_graph(groups, dfs, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Accuracy X Generation {seeds}')
    )

    for group in groups:
        df = dfs.get(group)
        add_line(fg, x=df.gen, y=df.specialist_score, name=f'Accuracy - group [{group}]')

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Accuracy')
    fg.show()

def precision_recall_group_graph(dfs, groups, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Precision/Recall X Generation {seeds}')
    )

    for group in groups:
        df = dfs.get(group)
        add_line(fg, x=df.gen, y=df.precision, name=f'Precision - group {group}')
        add_line(fg, x=df.gen, y=df.recall, name=f'Recall - group {group}')

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Precision/Recall')
    fg.show()

def score_graph(df, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Accuracy X Generation {seeds}')
    )

    add_line(fg, x=df.gen, y=df.specialist_score, name='Accuracy')

    fg.update_xaxes(title_text='Generation')
    fg.update_yaxes(title_text='Accuracy')
    fg.show()

def precision_recall_graph(df, seeds=''):
    fg = go.Figure(
        layout=go.Layout(title=f'Specialist Precision/Recall X Generation {seeds}')
    )

    add_line(fg, x=df.gen, y=df.precision, name='Precision')
    add_line(fg, x=df.gen, y=df.recall, name='Recall')

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
