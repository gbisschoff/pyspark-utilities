def as_list(x):
    if isinstance(x, list): return x
    if x is None: return []
    else: return [x]