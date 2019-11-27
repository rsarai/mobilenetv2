"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras import backend as K
from keras_octave_conv import OctaveConv2D

# from octave_mobilenetv2 import octave_conv
# from octave_conv_block import initial_oct_conv_bn_relu, final_oct_conv_bn_relu, oct_conv_bn_relu


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """
    high, low = inputs
    cchannel = int(filters * alpha)

    if not r:
        skip_high = Conv2D(int(filters * (1-alpha)), 1)(high)
        skip_high = BatchNormalization()(skip_high)
        skip_high = Activation(relu6)(skip_high)

        skip_low = Conv2D(int(filters * alpha), 1)(low)
        skip_low = BatchNormalization()(skip_low)
        skip_low = Activation(relu6)(skip_low)
    else:
        skip_high, skip_low = high, low

    high, low = OctaveConv2D(filters=filters, kernel_size=(3, 3))([high, low])
    high = BatchNormalization()(high)
    high = Activation(relu6)(high)
    low = BatchNormalization()(low)
    low = Activation(relu6)(low)

    # high, low = OctaveConv2D(filters=filters, kernel_size=(3, 3), strides=(s, s))([high, low])
    # high = BatchNormalization()(high)
    # high = Activation(relu6)(high)
    # low = BatchNormalization()(low)
    # low = Activation(relu6)(low)

    if r:
        high = Add()([high, skip_high])
        low = Add()([low, skip_low])

    return high, low


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """
    high, low = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        high, low = _bottleneck([high, low], filters, kernel, t, alpha, 1, True)

    return high, low


def OctaveMobileNetv2(input_shape, k, alpha=1):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].

    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)
    normal = BatchNormalization()(inputs)
    first_filters = _make_divisible(32 * alpha, 8)
    high, low = OctaveConv2D(first_filters, (3, 3), strides=2)(inputs)

    high, low = _inverted_residual_block([high, low], 16, (3, 3), t=6, alpha=alpha, strides=1, n=1)
    high, low = _inverted_residual_block([high, low], 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    high, low = _inverted_residual_block([high, low], 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    high, low = _inverted_residual_block([high, low], 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    high, low = _inverted_residual_block([high, low], 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    high, low = _inverted_residual_block([high, low], 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    high, low = _inverted_residual_block([high, low], 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    # high, low = MaxPool2D()(high), MaxPool2D()(low)
    conv = OctaveConv2D(last_filters, kernel_size=1, ratio_out=0.0)([high, low])
    # high_conv = _conv_block(high, last_filters, kernel=(1, 1), strides=(1, 1))
    # low_conv = _conv_block(low, last_filters, kernel=(1, 1), strides=(1, 1))
    # conv = layers.Add()([high_to_high, low_to_high])
    flatten = Flatten()(conv)
    normal = BatchNormalization()(flatten)
    dropout = Dropout(rate=0.4)(normal)
    outputs = Dense(units=10, activation='softmax')(dropout)

    model = Model(inputs, outputs)
    return model


if __name__ == '__main__':
    model = OctaveMobileNetv2((28, 28, 1), 100, 1.0)
    print(model.summary())
