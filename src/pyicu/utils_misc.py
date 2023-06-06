import multiprocessing
import pandas as pd

# Apply a function fun to columns (cols) in data frame x, ply_fun parameter specifies type of parallel processing
def col_ply(x, cols, fun, ply_fun=multiprocessing.Pool().map, *args, **kwargs):
    results = ply_fun(lambda y: fun(x[y], *args, **kwargs), cols)
    print("TEST")
    return pd.Series(results, index=cols)