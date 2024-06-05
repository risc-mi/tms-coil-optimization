import tensorflow as tf
import pickle
import gzip
import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation

def is_any_string_in_list_present(string, string_list):
    """Checks if a string is present in a list (case-insensitive).

    Args:
        string: The string to search for.
        string_list: The list of strings to search in.

    Returns:
        True if the string is found in the list (case-insensitive), False otherwise.
    """
    for item in string_list:
        if str(item) in str(string):
            return True
    return False


def convert_to_cond(ft):
    """Converts tissue type labels to conductivity values.

        Args:
        ft: Tissue type labels.

    Returns:
        ft: Corresponding conductivity values.
    """
    ft[ft==1] = 0.126 #wm
    ft[ft==2] = 0.275 #gm
    ft[ft==3] = 1.654 #csf
    ft[ft==4] = 0.01 #bone
    ft[ft==5] = 0.465 #scalp
    ft[ft==6] = 0.5 #eye balls
    ft[ft==7] = 0.008 #compact bone
    ft[ft==8] = 0.025 #spongy bone
    ft[ft==9] = 0.6 #blood
    ft[ft==10] = 0.16 #muscle
    ft[ft==51] = 0.8 #necrotic tumor core (NCR)
    ft[ft==52] = 0.2 #peritumoral edematous/infiltrated tissue (ED)
    ft[ft==54] = 0.4 #enhancing tumor (ET)

    return ft


def sine_loss(y_true, y_pred):
    """Calculates the sine loss between true and predicted values.

    Args:
        y_true: A TensorFlow tensor with the ground truth values.
        y_pred: A TensorFlow tensor with the predicted values.

    Returns:
        A TensorFlow scalar representing the mean absolute value of the sine of the difference.
    """
    diff = (y_true - y_pred) / 2.
    return tf.reduce_mean(tf.abs(tf.math.sin(diff)))


def load_data(path_, resample_=False):
    """
    Loads and processes .npy.gz files from the specified directory.

    Args:
        path (str): Path to the directory containing .npy.gz files.
        resample (bool): Whether to resample the data by downsampling and padding.

    Returns:
        dict: Dictionary containing loaded data with filenames (without extensions) as keys.
    """
    print('Start loading ' + str(path_))
    dict_data = {}
    files = [f for f in os.listdir(path_) if f.endswith('.npy.gz')]
    for f_ in files:
        with gzip.open(os.path.join(path_, f_), 'rb') as f:
            temp = np.transpose(np.load(f)).astype('float16')
            if resample_:
                temp = np.pad(temp[::2, ::2, ::2], [(4,4), (4,4), (0,0)], mode='constant')
        dict_data[f_.split('.npy.gz')[0]] = np.expand_dims(temp, axis=-1)

    print('Done loading ' + str(path_))

    return dict_data


def data_generator(batch_size, data_dir, file_list, t1s, tissues, dof, affine):
    """
    A generator function to yield batches of data for training.

    Args:
        batch_size (int): The number of samples in each batch.
        data_dir (str): Directory where data files are stored.
        file_list (list): List of filenames (without extensions) to be used for loading data.
        t1s (dict): Dictionary mapping subject IDs to their T1 maps.
        tissues (dict): Dictionary mapping subject IDs to their tissue conductivity maps.
        dof (dict): Dictionary mapping file names to degrees of freedom data for coil positioning.
        affine (dict): Dictionary mapping file names to affine matrices for coil orientation.

    Yields:
        tuple: A tuple containing a dictionary of input data and a dictionary of output data.
    """
    while True:
        # Shuffle the file list for each epoch
        np.random.shuffle(file_list)

        # Loop over batches of files
        for i in range(0, len(file_list), batch_size):
            # Initialize empty lists to hold the input   output data for the current batch
            batch_x1 = []
            batch_y1 = []
            batch_y2 = []
            batch_y3 = []

            # Loop over the files in the current batch
            for j in range(i, min(i+batch_size, len(file_list))):

                # get the current subject id
                subject_id = file_list[j].split('_id')[0]

                # Load the current file
                # last dimension has shape of 2:
                # - first is the target distancefield
                # - second is the induced electric field 
                targetdistance_efield_arr = np.load(os.path.join(data_dir, file_list[j] + '.npy'))

                # create the neural network's input
                # last dimension has shape of 3:
                # - first is the subject t1 map
                # - first is the subject tissue conductivity map
                # - third is the target distancefield
                x1_ = t1s[subject_id]
                x1_ = np.concatenate((x1_, convert_to_cond(tissues[subject_id])), axis=-1)                
                x1_ = np.concatenate((x1_, targetdistance_efield_arr[...,0:1]), axis=-1)
                
                # create the neural network's output 1: the induced electric field map
                y1_ = targetdistance_efield_arr[...,1:2]

                # create the neural network's output 2: the optimal coil positioning
                y2_ = dof[file_list[j]][:3]

                # create the neural network's output 3: the optimal coil orientation
                m_aff = affine[file_list[j]][:3,:3,0].T
                R = Rotation.from_matrix(m_aff)
                y3_ = R.as_euler('zxy')

                batch_x1.append(x1_)
                batch_y1.append(y1_)
                batch_y2.append(y2_)
                batch_y3.append(y3_)

            # Convert the batch lists to numpy arrays
            batch_x1 = np.array(batch_x1)
            batch_y1 = np.array(batch_y1)
            batch_y2 = np.array(batch_y2)
            batch_y3 = np.array(batch_y3)

            # Yield the current batch
            yield {'inputs': batch_x1}, {'output1': batch_y1, 'output2': batch_y2, 'output3': batch_y3} 
        
        
def save_to_gzip(outpath, outdata):
    """
    Saves data to a compressed file using the gzip format.

    Args:
        outpath (str): The output file path (including ".gz" extension).
        outdata (numpy.ndarray): The data to be saved.

    Raises:
        RuntimeError: If saving the data fails.
    """
    with gzip.open(outpath, 'wb') as f:
        np.save(f, outdata)