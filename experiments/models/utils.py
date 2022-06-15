import numpy as np
import pandas as pd

def column_group_mean(group):
    if len(group) > 1:
        m = np.mean(group, axis=0)
        return m[~np.isnan(m)]
    else:
        return group[0]

def accuracy(tp, tn, fn, fp):
    return (tp+tn)/(tp+fp+fn+tn)

def precision(tp, fp):
    return tp/(tp+fp)

def recall(tp, fn):
    return tp/(tp+fn)

def f1_score(p, r):
    return 2*(p*r)/(p+r)

def process_cm_metrics(dataset, columns):
    cm_data = dict(
        gen=dataset.gen,
        accuracy=[],
        precision=[],
        recall=[],
        f1_score=[]
    )

    for i in dataset.index:
        df = dataset.iloc[i]
        tp, fp, fn, tn = [df[col] for col in columns]
        a = accuracy(tp, tn, fn, fp)
        cm_data['accuracy'].append(a)
        p = precision(tp, fp)
        cm_data['precision'].append(p)
        r = recall(tp, fn)
        cm_data['recall'].append(r)
        f = f1_score(p, r)
        cm_data['f1_score'].append(f)

    return pd.DataFrame(cm_data)
