__all__ = [
    'clip',
    'repeat',
]


def clip(value, min_value=None, max_value=None):
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def repeat(obj, times: int):
    ix = 0
    while True:
        if times is None:
            pass
        elif ix < times:
            ix += 1
        else:
            break
        yield obj
