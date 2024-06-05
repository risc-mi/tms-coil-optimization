import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

import numpy as np
import os
from model import unet
from utils import is_any_string_in_list_present
from utils import (
    is_any_string_in_list_present,
    convert_to_cond,
    sine_loss,
    load_data,
    data_generator,
    save_to_gzip
)

# Define the model architecture
input_shape = (112, 96, 80, 3)
model = unet(input_shape=input_shape, initial_filters=32)

losses = {
    'output1': 'mean_squared_error',
    'output2': 'mean_absolute_error',
    'output3': sine_loss,
}
loss_weights = {
    'output1': 1,
    'output2': 0.002,
    'output3': 0.035,
}

# Compile the model with loss and Adam optimizer
model.compile(loss=losses, loss_weights=loss_weights, optimizer=Adam(learning_rate=5e-5))

# Print the model summary
model.summary()


path_data = './sample_data'
t1s = load_data(os.path.join(path_data, 't1'), resample_=True)
tissues = load_data(os.path.join(path_data, 'tissues'), resample_=True)
dof = load_data(os.path.join(path_data, 'dof'))

# affine matrix for the optimized coil positioning
# see also: https://simnibs.github.io/simnibs/build/html/documentation/sim_struct/position.html
# see also: https://simnibs.github.io/simnibs/build/html/documentation/opt_struct/tmsoptimize.html#tmsoptimize-doc
affine = load_data(os.path.join(path_data, 'affine_matrix'))
targetdistance_efield_path = os.path.join(path_data, 'targetdistance_efield') 

train_subjs = [ 100206 ]
val_subjs = [ 100206 ]
test_subjs = [ 100206 ]


data_files = [s.split('.npy')[0] for s in os.listdir(targetdistance_efield_path)]
num_files = len(data_files)
print("num_files:", num_files)

train_files = [f for f in data_files if is_any_string_in_list_present(f, train_subjs)]
num_train_files = len(train_files)
print("num_train_files:", num_train_files)

val_files = [f for f in data_files if is_any_string_in_list_present(f, val_subjs)]
num_val_files = len(val_files)
print("num_val_files:", num_val_files)

test_files = [f for f in data_files if is_any_string_in_list_present(f, test_subjs)]
num_test_files = len(test_files)
print("num_test_files:", num_test_files)

batch_size = 4
steps_per_epoch = num_train_files // batch_size
val_steps = num_val_files // batch_size

# Define the Checkpoint callback
best_checkpoint_path = './chkpts/model_best_chkpt.h5'
all_checkpoint_path = './chkpts/model_all_chkpt_epoch{epoch:02d}-val_los{val_loss:.5f}.h5'
mcp_save = ModelCheckpoint(all_checkpoint_path, save_best_only=False, monitor='val_loss', mode='min', save_freq='epoch')
mcp_best_save = ModelCheckpoint(best_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# Define ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=15, mode='min', min_lr=1e-6)

# Define CSV callback
csv_logger = CSVLogger('training.log', separator=',', append=False)

# Training
do_training = True
if do_training:
    print('Start training')
    model.fit(data_generator(batch_size, data_dir=targetdistance_efield_path, file_list=train_files, t1s=t1s, tissues=tissues, dof=dof, affine=affine), steps_per_epoch=steps_per_epoch,
              validation_data=data_generator(batch_size, data_dir=targetdistance_efield_path, file_list=val_files, t1s=t1s, tissues=tissues, dof=dof, affine=affine), validation_steps=val_steps,
              callbacks=[mcp_save, mcp_best_save, reduce_lr, csv_logger], epochs=25)
    print('End Training')
# After training, set the network weights to the best checkpoint (calculated on the validation set)
model.load_weights(best_checkpoint_path)

# Prediction
do_prediction = True
predictions_dir = './predictions'
if do_prediction:
    print('Start Prediction')
    for file_ in test_files:

        # filename for current prediction
        outfilename_ = file_ + '.npy.gz'

        # get the current subject id
        subject_id = file_.split('_id')[0]
        
        # Load the current file
        # last dimension has shape of 2:
        # - first is the target distancefield
        # - second is the induced electric field 
        targetdistance_efield_arr = np.load(os.path.join(targetdistance_efield_path, file_ + '.npy'))
           
        x1_ = t1s[subject_id]
        x1_ = np.concatenate((x1_, convert_to_cond(tissues[subject_id])), axis=-1)
        x1_ = np.concatenate((x1_, np.expand_dims(targetdistance_efield_arr[...,0], axis=-1)), axis=-1)
        x1_ = np.expand_dims(x1_, axis=0)
        
        prediction_ = model.predict(x1_)

        # Save emagn gt and prediction
        gt1_ = targetdistance_efield_arr[..., 1:2]
        outdata_ = np.concatenate((gt1_, prediction_[0][0, ...]), axis=-1)
        outpath_ = os.path.join(predictions_dir, 'emagn', outfilename_)
        save_to_gzip(outpath_, outdata_)
        
        # Save trans gt and prediction
        gt2_ = dof[file_][:3]
        outdata_ = np.concatenate((gt2_, prediction_[1].T), axis=-1)
        outpath_ = os.path.join(predictions_dir, 'trans', outfilename_)
        save_to_gzip(outpath_, outdata_)

        # Save rot gt and prediction
        from scipy.spatial.transform import Rotation
        m_aff = Rotation.from_matrix(affine[file_][:3, :3, 0].T)
        gt3_ = m_aff.as_euler('zxy')
        outdata_ = np.concatenate((gt3_.reshape((3, 1)), prediction_[2].T), axis=-1)
        outpath_ = os.path.join(predictions_dir, 'rot', outfilename_)
        save_to_gzip(outpath_, outdata_)
    
    print('End Prediction')