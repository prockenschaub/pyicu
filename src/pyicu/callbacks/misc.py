from pyicu.container import pyICUTbl

def transform_fun(fun, *args, **kwargs):
    transf_args = list(args)
    transf_kwargs = dict(kwargs)

    def callable(x: pyICUTbl, val_var=None, *args, **kwargs):
        # TODO: this currently changes values by reference. is that okay or do we need to copy?
        if val_var is None:
            val_var = x.data_var
        x[val_var] = fun(x[val_var], *transf_args, **transf_kwargs)
        return x
    
    return callable
