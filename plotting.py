## Functions and classes for plotting with different number of figures and kwargs
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Palatino",
})

def myplot(ydata, xdata=[], labels=[], title=''):
    """
    Creates a single plot/figure.
    Example use: myplot(x,y, [r'time since $\tau$ = 1970', 'snow height'], r'Snow Height Evaluation')
    :param xdata:
    :param ydata:
    :param labels:
    :param title:
    :return:
    """
    #TODO: Make option for multiple plots in one figure
    fig = plt.figure()
    if ydata.ndim > 1 and len(xdata)==0:
        xdata = np.arange(ydata.shape[0])
        print('YASSSSS')
        # by default, if ndim is passed, plot() plots columns i.e. [:,col]
        plt.plot(ydata, linewidth=0.5)
    elif ydata.ndim > 1 and len(xdata) > 0:
        plt.plot(xdata, ydata, linewidth=0.5)
    else:
        if xdata.shape == 0:
            plt.plot(ydata, linewidth=0.5)
        plt.plot(xdata, ydata, linewidth=0.5)
        if len(labels) == 2:
            plt.xlabel(r'xlabel {}'.format(labels[0]))
            plt.ylabel(f'ylable {labels[1]}')
        elif len(labels) == 1:
            plt.ylabel(f'xlabel {labels[0]}')
        else:
            pass
    plt.title(title)
    return fig


def mulplot():
    """
    Plotting with multiple axes and subplots
    :return:
    """
    pass
