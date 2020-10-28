import matplotlib.pyplot as plt


def set_color_map(cmap_name, n, ax):
    cm = plt.get_cmap(cmap_name)
    ax.set_prop_cycle(color=[cm(1. * i / n) for i in range(n)])
