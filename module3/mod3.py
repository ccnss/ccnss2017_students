from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

def ols(x, y, p=1):

    X = np.empty((len(x), p+1))

    for i in range(p+1): 
        X[:,i] = x**i

    Y = np.reshape(y, (x.shape[0], 1))

    w = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y))

    return w


def plot_data(x, y, xlim=None, ylim=None, title=None, ax_labels=None):
    
    x_min, x_max = (min(x), max(x))
    y_min, y_max = (min(y), max(y))
    x_delta, y_delta = (0.1*abs(x_max-x_min), 0.1*abs(x_max-x_min))
    
    if xlim is None:
        xlim = (x_min-x_delta, x_max+x_delta)
    if ylim is None:
        ylim = (y_min-y_delta, y_max+y_delta)
        
    if title is None:
        title = 'Data samples'
    if ax_labels is None:
        ax_labels = ('x','y')
        
    plt.plot(x, y, 'ko', alpha=0.2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    
    return


def plot_loss(loss, epochs, title=None, ax_labels=None):

    if title is None:
        title = 'Train loss'
    if ax_labels is None:
        ax_labels = ('step n','loss')
        
    plt.plot(range(len(loss)), loss, 'C0', linewidth=2)
    plt.xlim([-epochs*0.1, epochs*1.1])
    plt.title(title)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    
    return


def plot_loss_contour(x, y, w, f_loss, w_range=None, step=0.05,  title=None, ax_labels=None):

    (w0, w1) = w

    if w_range is None:
        w_range = (None, None)

    (w0_range, w1_range) = w_range

    if w0_range is None:
        w0_range = np.arange(w0-2, w0+2, step)

    if w1_range is None:
        w1_range = np.arange(w1-2, w1+2, step)

    if title is None:
        title = 'Parameters'

    if ax_labels is None:
        ax_labels = ('w0','w1')

    W0, W1 = np.meshgrid(w0_range, w1_range)
    Z = np.array([f_loss(x, y, (c0, c1)) for c0, c1 in zip(W0.flatten(), W1.flatten())]).reshape(W0.shape)
    levels = np.exp(np.arange(0, np.log(Z.max()), 0.5))

    plt.contour(W0, W1, Z, levels)
    plt.colorbar(format='%.0f')
    plt.title(title)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])

    return


def plot_bars(y, title=None, ax_labels=None):

    if title is None:
        title = 'Bar plot'
    if ax_labels is None:
        ax_labels = ('x','y')

    width = 1/1.5
    plt.bar(range(len(y)), y, width, color="C0")
    plt.title(title)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])

    return


def plot_generated(img_orig, img_gen, img_shape=None, n_show=10, selected=None):
    
    plt.figure(figsize=(10, 3))
    
    if selected is None:
        selected = np.random.randint(0, high=len(img_orig), size=n_show)

    for idx, items  in enumerate(zip(img_orig[selected], img_gen[selected])):
        orig, gen = items
        if img_shape is None:
            img_orig = np.squeeze(orig)
            img_gen = np.squeeze(gen)
        else:
            img_orig = np.reshape(orig, img_shape)
            img_gen = np.reshape(gen, img_shape)
            
        plt.subplot(2, n_show, idx + 1)
        plt.imshow(img_orig)
        plt.gray()
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, n_show, idx + 1 + n_show)
        plt.imshow(img_gen)
        plt.gray()
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        
    plt.tight_layout()
    
    return

def plot_embedding(x, y, title=None):
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(x.shape[0]):
        plt.text(x[i, 0], x[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9},
                 alpha=0.8)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()
