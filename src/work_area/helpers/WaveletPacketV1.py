import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pywt import WaveletPacket2D as wp2d
import pywt
import cv2
import torch
from einops import rearrange, repeat

####################################################################################
#   Decompose the Image using Wavelet Packet Transform (Merge Features with Image)
####################################################################################
def wp_decompose(image, wavelet_fun, level, format="Stack_ImgUseSameSize"):
    # This function decomposes the input 2D image into (2**level)**2 2D features
    # hence if we have only one level the output 2D features will be 4
    # The 2D features could be arranged as a matrix of (2**level) rows and (2**level) cols
    # ranging from the most approximate feature at location (0,0) to the most detailed
    # feature at location ((2**level) - 1,(2**level) - 1)
    # The original 2D wave_packet_decompose function down samples the input by 2 after each filter
    # Note: The 2D wave_packet_decompose function deals with a single channel image only.
    # hence you need to take a single channel out of the 3 channel coloured image
    # There is different options to take single channel, one of them is to convert to gray.
    #
    # This function gives the option to output the final features in one of 2 shapes:
    # 1) "stacked": in 3D matrix of a shape:(feature_height x feature_width x no_of_features)
    # 2) not "stacked": i.e. tiled in a 2D matrix of shape:
    # ((feature_height x features_rows) x (feature_width x features_cols))
    # note that features_rows = features_cols
    #
    # The format switch can take one of three values:
    # "Tile": apply wavelet --> features with smaller size --> tile the features to form a matrix similar size to the image
    # "Stack_ImgUseSameSize": Up-size image --> apply wavelet --> features size = image size
    # "Stack_ImgUseFeatureSize": apply wavelet --> features with smaller size --> downsize original image to feature size
    # Finally concatenate features to original image

    img_h = image.shape[0]
    features_rows = 2**level

    if format == "Stack_ImgUseSameSize":
        if level == 1:
            new_size = (img_h-1)*features_rows
        else:
            new_size = (img_h-2)*features_rows
        assert new_size < 2048

        dim = (new_size, new_size)
        # Apply wavelet packet on a resized version of the image
        resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    elif format == "Tile":
        if level == 1:
            new_size = (img_h-2)
        else:
            new_size = (img_h-2*features_rows)

        dim = (new_size, new_size)
        # Apply wavelet packet on a resized version of the image
        resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = image

    img_one_channel = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # apply wavelet packet transform
    wp = wp2d(data=img_one_channel, wavelet=wavelet_fun, mode='symmetric')

    # get the paths of the image
    paths = [node.path for node in wp.get_level(level)]

    # Arrange the paths in a 2D matrix shape, useful to visualize the wavelet packet features
    # also needed to output to arrange the features in a tile format.
    paths_rows = []
    paths_matrix = []
    for i, path in enumerate(paths):
        if (i+1) % features_rows == 0:
            paths_rows.append(path)
            paths_matrix.append(paths_rows)
            paths_rows = []
        else:
            paths_rows.append(path)

    if format == "Tile":
        nodes_matrix = []
        for row in paths_matrix:
            nodes_rows = []
            # print(row)  # >>>>>>
            for element in row:
                nodes_rows.append(wp[element].data)
            nodes_matrix.append(nodes_rows)

        nodes_tiled = np.array(nodes_matrix)
        # print("nodes_tiled", nodes_tiled.shape)  # >>>>>>
        # print("total shape", nodes_tiled.shape[-1]*features_rows)  # >>>>>>
        nodes_arr = nodes_tiled.swapaxes(2, 1).reshape(img_h, img_h, 1)
        # print("nodes_arr", nodes_arr.shape)  # >>>>>>
        # print("nodes_arr", nodes_arr)  # >>>>>>
        axis = 2
    else:
        nodes = [wp[path].data for path in paths]
        nodes_arr = np.transpose(np.array(nodes), (1, 2, 0))
        axis = 2

    if format == "Stack_ImgUseFeatureSize":
        output_dim = wp[paths[0]].data.shape
        # print("output_dim", output_dim)  # >>>>>>
        output_image = cv2.resize(
            image, output_dim, interpolation=cv2.INTER_LINEAR)
    else:
        output_image = image

    # print("axis", axis) # >>>>>>

    img_and_nodes_array = np.concatenate((output_image, nodes_arr), axis=axis)

    ###############################################################################
    wp_fun = wp[paths[0]].wavelet.wavefun()
    # x, y = wp_fun[-1], wp_fun[0]
    wp_name = wp[paths[0]].wavelet.family_name

    return img_and_nodes_array, paths, resized_img, features_rows, wp_fun, wp_name


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
    wp = [wp2d(data=img_one_channel[i], wavelet=wavelet_fun, mode='symmetric') for i in range(3)]

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
    # nodes_arr = np.transpose(np.array(nodes), (1, 2, 0))
    # axis = 2

    # print("axis", axis) # >>>>>>

    # --->(16, feature_height, feature_width, 3)
    nodes_array = rearrange(np.array(nodes), "c f fh fw -> f fh fw c")

    # --->(16*3, feature_height, feature_width)
    nodes_tensor = torch.tensor(rearrange(np.array(nodes), "c f fh fw -> (c f) fh fw"))

    ###############################################################################
    wp_fun = wp[0][paths[0]].wavelet.wavefun()
    # # x, y = wp_fun[-1], wp_fun[0]
    wp_name = wp[0][paths[0]].wavelet.family_name

    return img, nodes_array, paths, features_rows, wp_fun, wp_name, nodes_tensor

####################################################################################
#                    Norm to Plot
####################################################################################
def norm_to_plot(img):

    TensorOrigin = False
    CHWOrigin = False

    # Make sure the used Image is numpy array with (Height,Width,Channel) shape
    if torch.is_tensor(img):
        TensorOrigin = True
        img = img.numpy()

    if img.shape[0] == 3:
        CHWOrigin = True
        img = rearrange(img, 'c h w -> h w c')

    scale = 1.0/max(img.max(), abs(img.min()))
    img_out = (img * scale)
    # img_out = (img_out+1)/2

    return img, img_out

####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_img_grid(holder, data, main_title, sub_title=None, Grid2D=True, axes_pad=0.3):

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
        ax.imshow(data[r, c])
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
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_wpt_nodes(image, wavelet_fun, level):
    img, nodes, paths, rows, *_ = wpt_dec(image, wavelet_fun, level)
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams["figure.autolayout"] = True

    fig = plt.figure(figsize=(12, 12))
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1.5], hspace=0.1, squeeze='True')

    axs0 = subfigs[0].subplots(1, 1)
    axs0.imshow(norm_to_plot(img))
    # axs0.set_xticks([])
    # axs0.set_yticks([])
    axs0.set_title("Image")

    grid_text = "Features extracted using Wavelet Packet Transform"

    nodes_grid = np.reshape(nodes[:, :, :, 0],(int(np.sqrt(nodes.shape[0])),-1,nodes.shape[1],nodes.shape[2]))

    plot_img_grid(subfigs[1], nodes_grid, grid_text, paths, Grid2D=True)

####################################################################################
#                    Plot the Image and its Extracted WPT Features
####################################################################################
def plot_wp_nodes(image, wavelet_fun, level, format):
    nodes, paths, _, rows, *_ = wp_decompose(image, wavelet_fun, level, format)

    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams["figure.autolayout"] = True

    fig = plt.figure(figsize=(12, 12))
    subfigs = fig.subfigures(2, 1, height_ratios=[
                             1, 1.5], hspace=0.1, squeeze='True')


    axs1 = subfigs[0].subplots(1, 1)
    axs1.imshow(nodes[:, :, :3].astype(np.uint8))
    axs1.set_title("Adjusted Image")

    if format != "Tile":
        grid = ImageGrid(subfigs[1], 111, nrows_ncols=(
            rows, rows), axes_pad=0.3)

        for i, ax in enumerate(grid):

            ax.imshow(nodes[:, :, i+3])
            ax.set_title(paths[i])
        features_text = "Features extracted using Wavelet Packet Transform"
        subfigs[1].suptitle(features_text)
    else:

        axs2 = subfigs[1].subplots(1, 1)
        axs2.imshow(nodes[:, :, 3])
        axs2.set_title("Tiled Features")


####################################################################################
#                     Plot the Wavelet Impulse function
####################################################################################
def plot_wp_fun(image, wavelet_fun, level, format):
    *_, wp_fun, wp_name = wp_decompose(image, wavelet_fun, level, format)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')

    axs[0].plot(wp_fun[-1], wp_fun[0])
    axs[0].grid(True)
    axs[0].set_title(f'{wavelet_fun}')

    w = pywt.Wavelet(wavelet_fun)
    (phi, psi, x) = w.wavefun(level=level)

    axs[1].plot(x, phi)
    axs[1].grid(True)
    axs[1].set_title(f'{wavelet_fun} phi')

    axs[2].plot(x, psi)
    axs[2].grid(True)
    axs[2].set_title(f'{wavelet_fun} psi')


####################################################################################
#                     Plot the Wavelet Impulse function
####################################################################################
def plot_wpt_fun(image, wavelet_fun, level):
    *_,wp_fun, wp_name,_ = wpt_dec(image, wavelet_fun, level)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')

    axs[0].plot(wp_fun[-1], wp_fun[0])
    axs[0].grid(True)
    axs[0].set_title(f'{wavelet_fun}')

    w = pywt.Wavelet(wavelet_fun)
    (phi, psi, x) = w.wavefun(level=level)

    axs[1].plot(x, phi)
    axs[1].grid(True)
    axs[1].set_title(f'{wavelet_fun} phi')

    axs[2].plot(x, psi)
    axs[2].grid(True)
    axs[2].set_title(f'{wavelet_fun} psi')