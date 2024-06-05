import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
    GlobalMaxPooling3D,
    Dropout,
    Activation,
    Multiply,
    SpatialDropout3D,
    Dense,
)
from tensorflow.keras.activations import gelu
from tensorflow.keras.regularizers import L1L2


def attention_gate(g, s, filters):
    """
    Attention gate mechanism to focus on relevant features.

    Args:
        g: Input feature tensor.
        s: Skip connection feature tensor.
        filters: Number of filters for the attention layer.

    Returns:
        The input feature tensor weighted by the attention coefficients.
    """
    # Compute attention coefficients
    Wg = Conv3D(filters, kernel_size=1, padding="same", kernel_regularizer=L1L2(l1=0, l2=0))(g)    
    Ws = Conv3D(filters, kernel_size=1, padding="same", kernel_regularizer=L1L2(l1=0, l2=0))(s)
    
    combined = gelu(tf.add(Wg, Ws))
    combined = Conv3D(filters, kernel_size=1, padding="same", kernel_regularizer=L1L2(l1=0, l2=0))(combined)

    attention = Activation('sigmoid')(combined)

    # Apply attention to the input features
    return Multiply()([attention, s])
    
def unet(input_shape, initial_filters=32):
    """
    Creates a U-Net architecture with attention gates.

    Args:
        input_shape: Shape of the input data (channels, height, width, depth).
        initial_filters: Number of filters for the first convolutional layer.

    Returns:
        A Keras model.
    """
    inputs = Input(shape=input_shape, name='inputs')

    # Encoding Path
    conv1 = Conv3D(initial_filters, (7, 7, 7), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(inputs)
    conv1 = gelu(conv1)
    conv1 = Conv3D(initial_filters, (7, 7, 7), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv1)
    conv1 = gelu(conv1)
    conv1 = SpatialDropout3D(0.45)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    filters = initial_filters*2
    conv2 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(pool1)
    conv2 = gelu(conv2)
    conv2 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv2)
    conv2 = gelu(conv2)
    conv2 = SpatialDropout3D(0.45)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    filters = initial_filters*4
    conv3 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(pool2)
    conv3 = gelu(conv3)
    conv3 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv3)
    conv3 = gelu(conv3)
    conv3 = SpatialDropout3D(0.45)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bottleneck
    filters = initial_filters*8
    conv4 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(pool3)
    conv4 = gelu(conv4)
    conv4 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv4)
    conv4 = gelu(conv4)
    conv4 = SpatialDropout3D(0.45)(conv4)

    # Decoding Path
    filters = initial_filters*4
    up7 = UpSampling3D(size=(2, 2, 2))(conv4)
    up7 = Conv3D(filters, (2, 2, 2), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(up7)
    up7 = gelu(up7)
    s = attention_gate(up7, conv3, filters)
    merge7 = concatenate([up7, s], axis=4)
    conv7 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(merge7)
    conv7 = gelu(conv7)
    conv7 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv7)
    conv7 = gelu(conv7)
    conv7 = SpatialDropout3D(0.45)(conv7)

    filters = initial_filters*2
    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    up8 = Conv3D(filters, (2, 2, 2),  padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(up8)
    up8 = gelu(up8)
    s = attention_gate(up8, conv2, filters)    
    merge8 = concatenate([up8, s], axis=4)
    conv8 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(merge8)
    conv8 = gelu(conv8)
    conv8 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv8)
    conv8 = gelu(conv8)
    conv8 = SpatialDropout3D(0.45)(conv8)

    filters = initial_filters
    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    up9 = Conv3D(filters, (2, 2, 2), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(up9)
    up9 = gelu(up9)
    s = attention_gate(up9, conv1, filters)    
    merge9 = concatenate([up9, s], axis=4)
    conv9 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(merge9)
    conv9 = gelu(conv9)
    conv9 = Conv3D(filters, (3, 3, 3), padding='same', kernel_regularizer=L1L2(l1=0, l2=0))(conv9)
    conv9 = gelu(conv9)
    conv9 = SpatialDropout3D(0.45)(conv9)
    
    # Output1 - emagn
    output1 = Conv3D(1, (1, 1, 1), activation='linear', dtype=tf.float32, name='output1')(conv9)

    # Output2&3 - trans & rot
    input_dense = GlobalMaxPooling3D()(conv4)
    input_dense = Dropout(0.75)(input_dense)
    output2 = Dense(3, activation='linear', dtype=tf.float32, name='output2')(input_dense)
    output3 = Dense(3, activation='linear', dtype=tf.float32, name='output3')(input_dense)
    
    # Create the model
    model = Model(inputs=[inputs], outputs=[output1, output2, output3])

    return model