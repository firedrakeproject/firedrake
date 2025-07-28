from firedrake.device import device


def on_host(func):
    def wrapper(*args, **kwargs):
        with device("cpu") as compute_device:
            res = func(*args, **kwargs)
        return res
    return wrapper

