import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from einops import rearrange


####################################################################################
#                                  Norm to Plot
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
#                        Plot the Image and its Patches
####################################################################################

def img_plot(plot, img, slice_width, CHW_Image=True, figsize=(8, 5), setticks1=None, setticks2=None, axes_pad=0.15):
    # img is passed as torch tensor, hence we need to arrange its shape to suite imshow

    img_a = norm_to_plot(img)[1]

    img_patches_matrix = rearrange(
        img_a, '(row h) (col w) c -> row col h w c', h=slice_width, w=slice_width)

    print("Shape of the image patches matrix: ", img_patches_matrix.shape)

    grid1_text = "Input Image Sliced to Patches"
    grid2_text = "Image Patches Arranged Sequentially Before Enter to Encoder"

    # plt.rcParams['figure.constrained_layout.use'] = True
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 2], hspace=0.1, squeeze='True')

    ax0 = subfigs[0].subplots(1, 1)
    # ax0.imshow(img_a)
    ax0.imshow(img_a)

    if setticks1 is not None:
        ax0.set(xticks=np.arange(0, img.shape[1]+1, step=setticks1),
                yticks=np.arange(0, img.shape[0]+1, step=setticks1))
    else:
        ax0.set(xticks=[], yticks=[])

    ax0.set_title("Input Image")

    plot_img_grid(subfigs[1], img_patches_matrix,
                  grid1_text, Grid2D=True, setticks=setticks2, axes_pad=axes_pad)

    # plot_img_grid(subfigs[2], img_patches_matrix,
    #               grid2_text, Grid2D=False, setticks=setticks2, axes_pad=axes_pad)
