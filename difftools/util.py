import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def set_color_map(cmap_name, n, ax):
    cm = plt.get_cmap(cmap_name)
    ax.set_prop_cycle(color=[cm(1. * i / n) for i in range(n)])

def dump_obj(dump_path, obj, name):
    with open(Path(dump_path, name), 'wb') as f:
        pickle.dump(obj, f)

def load_obj(dump_path, name):
    obj = None
    with open(Path(dump_path, name), mode="rb") as f:
        obj = pickle.load(f)
    return obj
