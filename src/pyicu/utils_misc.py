'''
Utility functions
=================

Several utility functions exported for convenience.
'''
#import multiprocessing
#import pandas as pd
#import numpy as np

    
# Apply a function fun to columns (cols) in data frame x, ply_fun parameter specifies type of parallel processing
#def col_ply(x, cols, fun, ply_fun=multiprocessing.Pool().map, *args, **kwargs):
#    results = ply_fun(lambda y: fun(x[y], *args, **kwargs), cols)
#    print("TEST")
#    return pd.Series(results, index=cols)

def chr_ply(x, fun, length=1, use_names=False, *args, **kwargs):
    return [str(fun(item, *args, **kwargs)) for item in x]