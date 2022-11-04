from typing import Any

def enlist(x: Any):
    # TODO: Test for scalar instead
    if not isinstance(x, list):
        return [x]
    else:
        return x