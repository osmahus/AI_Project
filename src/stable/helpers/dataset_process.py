# import cudf.pandas
# cudf.pandas.install()
import os
from torchvision.io import read_image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder


def dataset_data(img_path, folder_name, img_class):
    fname = os.listdir(img_path + "/"+folder_name)
    fname.sort()
    fpath = [img_path + "/"+folder_name+"/" + f for f in fname]
    sizes = [read_image(f).size() for f in fpath]
    labels = [img_class]*len(fname)

    return fpath, sizes, labels

# Define a search function


def search_string(s, search):
    return search in str(s).lower()

# ---------------------------------------------------------------
def onehot(label,classes=2):
    onehot_val=str(np.base_repr(2**label, 2)).zfill(classes)
    onehot_array=np.array([int(i)for i in [*onehot_val]])
    return onehot_array

# ---------------------------------------------------------------

def remove_fname_space(path):
    for filename in os.listdir(path):
        my_source = path + "/" + filename
        my_dest = path + "/" + filename.strip().replace(" ", "")
        os.rename(my_source, my_dest)


def dataset_to_df(absolute_path, relative_paths, paths_classes, train_ratio, val_ratio, test_ratio):
    classes = list(set(paths_classes))
    classes.sort(reverse=True)

    fpath = [""]*len(paths_classes)
    sizes = [""]*len(paths_classes)
    labels = [""]*len(paths_classes)

    fpath_total = []
    sizes_total = []
    labels_total = []

    for i in range(len(paths_classes)):
        fpath[i],  sizes[i], labels[i] = dataset_data(
            absolute_path, relative_paths[i], paths_classes[i])
        fpath_total += fpath[i]
        sizes_total += sizes[i]
        labels_total += labels[i]

    feature_col = "Image_path"
    sizes_col = "Image_size"
    cat_class_col = "Class"
    num_class_col = "Class_Codes"
    onehot_class_col="Class_Onehot"

    dict_all = {feature_col: fpath_total,sizes_col: sizes_total, cat_class_col: labels_total}
    df_all = pd.DataFrame(dict_all)
    df_all[cat_class_col] = df_all[cat_class_col].astype('category')
    df_all[num_class_col] = df_all[cat_class_col].cat.codes
    df_all[onehot_class_col] = df_all[num_class_col].apply(lambda row: onehot(row))#, axis=1)
    

    val_rel_ratio = val_ratio/(val_ratio+test_ratio)

    df_train, df_tmp = train_test_split(
        df_all, train_size=train_ratio, stratify=df_all[[cat_class_col]])
    df_val, df_test = train_test_split(
        df_tmp, train_size=val_rel_ratio, stratify=df_tmp[[cat_class_col]])

    classes_stats = np.zeros((3, len(classes)), dtype=int)

    for i in range(len(classes)):
        # Qty of the class
        classes_stats[0, i] = len(
            df_train[df_train[cat_class_col] == classes[i]])
        classes_stats[1, i] = len(df_val[df_val[cat_class_col] == classes[i]])
        classes_stats[2, i] = len(
            df_test[df_test[cat_class_col] == classes[i]])

    classes_stats_df = pd.DataFrame(classes_stats.tolist(), columns=classes)
    classes_stats_df["Total"] = classes_stats_df.sum(axis=1)
    classes_stats_df.index = ['Training', 'Validation', 'Testing']
    classes_stats_df.loc["Row_Total"] = classes_stats_df.sum()

    return df_all, df_train, df_val, df_test, classes_stats_df


def search_df(ref_df, str):
    # Search for the string in all columns
    mask = ref_df.apply(lambda x: x.map(lambda s: search_string(s, str)))
    # Filter the DataFrame based on the mask
    filtered_df = ref_df.loc[mask.any(axis=1)]
    return filtered_df.index[0]
