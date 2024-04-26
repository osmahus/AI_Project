import numpy as np
import matplotlib.pyplot as plt
from pywt import WaveletPacket2D as wp2d
import pywt
import torch
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


####################################################################################
#                    Norm to Plot
####################################################################################
def norm_to_plot(img):

    TensorOrigin = False
    CHWOrigin = False
    # print(img.shape)  # >>>>>>>>>>>>>>>>
    # Make sure the used Image is numpy array with (Height,Width,Channel) shape
    if torch.is_tensor(img):
        TensorOrigin = True
        img = img.numpy()

    if img.shape[0] == 3:
        CHWOrigin = True
        img = rearrange(img, 'c h w -> h w c')

    img_out = (img-img.min())/(img.max()-img.min())
    # print(img_out.shape)  # >>>>>>>>>>>>>>>>

    return img, img_out


####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_img_grid(holder, data, main_title, sub_title=None, Grid2D=True, setticks=None, normalize=False, axes_pad=0.3):

    data_rows = data.shape[0]
    data_cols = data.shape[1]
    if Grid2D:
        nrows = data_rows
        ncols = data_cols
    else:
        nrows = 1
        ncols = data_rows*data_cols

    x_label = range(ncols)
    y_label = range(nrows)

    grid = ImageGrid(holder, 111, (nrows, ncols), axes_pad=axes_pad)

    for i, ax in enumerate(grid):
        r, c = i//data_cols, i % data_cols
        if normalize:
            ax.imshow(norm_to_plot(data[r, c])[1])
        else:
            ax.imshow(data[r, c])
        if setticks is not None:
            ax.set(xticks=np.arange(0, data[r, c].shape[1]+1, step=setticks),
                   yticks=np.arange(0, data[r, c].shape[0]+1, step=setticks))
        else:
            ax.set(xticks=[], yticks=[])

        if Grid2D:
            if r == nrows - 1:
                ax.set_xlabel(x_label[c], rotation=0, fontsize=10, labelpad=20)
            if c == 0:
                ax.set_ylabel(y_label[r], rotation=0, fontsize=10, labelpad=20)
        else:
            ax.set_xlabel(x_label[i], rotation=0, fontsize=10, labelpad=20)
            if i == 0:
                ax.set_ylabel(y_label[i], rotation=0, fontsize=10, labelpad=20)
        if sub_title is not None:
            ax.set_title(sub_title[i])

    holder.suptitle(main_title)


####################################################################################
#      Decompose the Image using Wavelet Packet Transform(Keep Features Only)
####################################################################################

def wpt_dec(img, wavelet_fun, level):
    # This function decomposes the input 2D image into (2**level)**2 2D features
    # hence if we have only one level the output 2D features will be 4
    # The 2D features could be arranged as a matrix of (2**level) rows and (2**level) cols
    # ranging from the most approximate feature at location (0,0) to the most detailed
    # feature at location ((2**level) - 1,(2**level) - 1)
    # The original 2D wave_packet_decompose function down samples the input by 2 after each filter
    # Note: The 2D wave_packet_decompose function deals with a single channel image only.
    # hence you need to take a single channel out of the 3 channel coloured image
    #
    # This function gives the option to output the final features in "stacked" 3D matrix
    # of a shape:
    # if numpy  : (no_of_features x feature_height x feature_width x 3 Channels)
    # if torch  : ((no_of_features x 3 Channels) x feature_height x feature_width)

    TensorOrigin = False
    CHWOrigin = False

    img_one_channel = [0]*3
    wp = [0]*3
    paths = [0]*3

    # Make sure the used Image is numpy array with (Height,Width,Channel) shape.
    if torch.is_tensor(img):
        TensorOrigin = True
        img = img.numpy()

    if img.shape[0] == 3:
        CHWOrigin = True
        img = rearrange(img, 'c h w -> h w c')

    img_h = img.shape[0]
    features_rows = 2**level

    img_one_channel = [img[:, :, i] for i in range(3)]

    # apply wavelet packet transform
    wp = [wp2d(data=img_one_channel[i], wavelet=wavelet_fun,
               mode='symmetric') for i in range(3)]

    # get the paths of the image
    paths = [node.path for node in wp[0].get_level(level)]

    # Arrange the paths in a 2D matrix shape, useful to visualize the wavelet packet features
    paths_rows = []
    paths_matrix = []
    for i, path in enumerate(paths):
        if (i+1) % features_rows == 0:
            paths_rows.append(path)
            paths_matrix.append(paths_rows)
            paths_rows = []
        else:
            paths_rows.append(path)

    nodes = [[wp[i][path].data for path in paths] for i in range(3)]
    node_shape = wp[0][paths[0]].data.shape

    # print("axis", axis) # >>>>>>

    # --->(16, feature_height, feature_width, 3)
    nodes_array = rearrange(np.array(nodes), "c f fh fw -> f fh fw c")

    # --->(16*3, feature_height, feature_width)
    nodes_tensor = torch.tensor(
        rearrange(np.array(nodes), "c f fh fw -> f c fh fw"))

    ###############################################################################
    wp_fun = wp[0][paths[0]].wavelet.wavefun()
    # # x, y = wp_fun[-1], wp_fun[0]
    wp_name = wp[0][paths[0]].wavelet.family_name

    return img, nodes_array, paths, features_rows, wp_fun, wp_name, node_shape, nodes_tensor


####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
# def plot_wpt_nodes(image, wavelet_fun, level, setticks1=None, setticks2=None):
#     img, nodes, paths, rows, *_ = wpt_dec(image, wavelet_fun, level)
#     plt.rcParams['figure.constrained_layout.use'] = True
#     # plt.rcParams["figure.autolayout"] = True

#     img_a = norm_to_plot(img)[1]

#     fig = plt.figure(figsize=(20, 20))
#     subfigs = fig.subfigures(2, 1, height_ratios=[
#                              1, 3], hspace=0.1, squeeze='True')

#     axs0 = subfigs[0].subplots(1, 1)
#     axs0.imshow(img_a)
#     if setticks1 is not None:
#         axs0.set(xticks=np.arange(0, img.shape[1]+1, step=setticks1),
#                  yticks=np.arange(0, img.shape[0]+1, step=setticks1))
#     else:
#         axs0.set(xticks=[], yticks=[])
#     axs0.set_title("Image")

#     grid_text = "Features extracted using Wavelet Packet Transform"

#     nodes_grid = np.reshape(nodes[:, :, :, :],
#                             (int(np.sqrt(nodes.shape[0])), -1,
#                              nodes.shape[1], nodes.shape[2], nodes.shape[3]))

#     plot_img_grid(subfigs[1], nodes_grid, grid_text,
#                   paths, Grid2D=True, normalize=True, setticks=setticks2)


####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_wpt_nodes(image, wavelet_fun, level, setticks1=None, setticks2=None, figsize=(10, 10)):
    img, nodes, paths, rows, *_ = wpt_dec(image, wavelet_fun, level)
    plt.rcParams['figure.constrained_layout.use'] = True

    img_a = norm_to_plot(img)[1]

    fig = plt.figure(figsize=figsize)

    grid_text = "Features extracted using Wavelet Packet Transform"

    nodes_grid = np.reshape(nodes[:, :, :, :],
                            (int(np.sqrt(nodes.shape[0])), -1,
                             nodes.shape[1], nodes.shape[2], nodes.shape[3]))

    plot_img_grid(fig, nodes_grid, grid_text,
                  paths, Grid2D=True, normalize=True, setticks=setticks2)


####################################################################################
#                     Plot the Wavelet Impulse function
####################################################################################
def plot_wpt_fun(image, wavelet_fun, level):
    *_, wp_fun, wp_name, _, _ = wpt_dec(image, wavelet_fun, level)

    fig, axs = plt.subplots(1, 1, figsize=(2, 2), layout='constrained')

    axs.plot(wp_fun[-1], wp_fun[0])
    axs.grid(True)
    axs.set_title(f'{wavelet_fun}')
