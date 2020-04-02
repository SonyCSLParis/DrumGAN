import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# black with variable alpha (0: no transparent, 1: full transparent)
mask_cdict = {'red':   [[0.0,  0.0, 0.0], # anchor point, left value, right value
                        [1.0,  0.0, 0.0]],
              'green': [[0.0,  0.0, 0.0],
                        [1.0,  0.0, 0.0]],
              'blue':  [[0.0,  0.0, 0.0],
                        [1.0,  0.0, 0.0]],
              'alpha': [[0.0,  1.0, 1.0], 
                        [1.0,  0.0, 0.0]]}
mask_cmap = LinearSegmentedColormap('mask', mask_cdict)

ee = {'red':   [[0.0,  0.0, 0.0], # anchor point, left value, right value
                        [1.0,  0.0, 0.0]],
              'green': [[0.0,  1.0, 1.0],
                        [1.0,  1.0, 1.0]],
              'blue':  [[0.0,  0.0, 0.0],
                        [1.0,  0.0, 0.0]],
              'alpha': [[0.0,  1.0, 1.0], 
                        [1.0,  0.0, 0.0]]}
mask_cmap_ee = LinearSegmentedColormap('maskee', ee)


def rain2graph(rainbowgram, ax=None, fig=None):
    """
    Plot rainbowgram
    Args:
        rainbowgrams ([(mag, IF)]): list of rainbowgram datum (tuple of power and IF)
        ax :
        fig: 
    """
    ax = ax if ax is not None else plt.subplot(1, 1, 1)

    # IF color plot
    arg_im = ax.matshow(rainbowgram[1][::-1, :], cmap="rainbow") # small index should be lower part
    # arg_im = ax.matshow(rainbowgram[1][::-1, :], cmap=mask_cmap_ee) # small index should be lower part
    # mask by magnitude
    mag_im = ax.matshow(rainbowgram[0][::-1, :], cmap=mask_cmap)

    return ax
    # mag_im = ax.matshow(rainbowgram[0][::-1, :], cmap="viridis")
    # _ = fig.colorbar(mag_im, ax=ax) if fig is not None else 0
    # _ = fig.colorbar(arg_im, ax=ax) if fig is not None else 0