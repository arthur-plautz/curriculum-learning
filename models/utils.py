import numpy as np

def column_group_mean(group):
    if len(group) > 1:
        m = np.mean(group, axis=0)
        return m[~np.isnan(m)]
    else:
        return group[0]
