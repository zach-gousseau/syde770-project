import math
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import LeakyReLU, Masking

import model_utilities as mu


def spatial_feature_pyramid_net_vectorized_ND(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Historical input data is encoded using an spatial pyramid network
        https://arxiv.org/pdf/1606.00915.pdf
        https://arxiv.org/pdf/1612.03144.pdf
        https://arxiv.org/pdf/1612.01105.pdf
    Once each input data has been independently encoded though the spatial feature pyramid,
        the historical input data is further encoded into a single feature-cube using a convolutional-LSTM with
        return_sequence=False.
        https://arxiv.org/pdf/1506.04214.pdf
        This tensor is also concatenated with the latest day of historical input data to
        preserve certain input features for which only the most recent day is important.
    From the merged encoded state, a sequence of output steps is produced using the custom ResStepDecoder layer
        ResStepDecoder may be described as:
            ResStepDecoder(inputEncodedState=E):
                S[0] = G(E)
                for i in (1, Sequence_Length) do:
                    S[i] := S[i-1] + F(E, S[i-1])

                return S[1:]

            Where G is a learned function to estimate an initial state from the encoded state E.
            Where F is s learned function to predict the delta between each subsequent time-step.

        ResStepDecoder makes use  of TensorFlow's highly optimized SeparableConv2D to increase computation efficiency
        and reduce the total number of parameters.

    Finally, the sequence of extrapolated states are converted into the ice-presence probability through a
        time-distributed network-in-a-network structure, where the final layer uses a Sigmoid activation function.

    For use with ensembles.

    Matthew King, November 2019 @ NRC
    :return: Keras model object
    """
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    if 'output_steps' in kwargs:
        output_steps = kwargs['output_steps']
        if type(output_steps) != int:
            raise TypeError('Received output_steps of non-int type')
    else:
        output_steps = 30

    if 'leaky_relu_alpha' in kwargs:
        alpha = kwargs['leaky_relu_alpha']
        if type(alpha) != float:
            raise TypeError('Received leaky relu alpha of non-float type')
    else:
        alpha = 0.01

    if 'debug' in kwargs:
        debug = kwargs['debug']
        if type(debug) != bool:
            raise TypeError('Received debug of non-bool type')
    else:
        debug = False

    inputs = tf.keras.Input(shape=input_shape)

    n_features = 24

    full_res_map = mu.spatial_feature_pyramid(inputs, n_features, (3, 3), 8, alpha=alpha, l2_rate=l2,
                                              return_all=False, debug=debug)

    encoded_state = kl.ConvLSTM2D(48-input_shape[-1], (3, 3), padding='same', activation='selu',
                              kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),
                              name='State_Encoder')(full_res_map)

    days_in = input_shape[0]
    input_last_day_only = kl.Cropping3D(cropping=((days_in-1, 0), (0, 0), (0, 0)))(inputs)
    input_last_day_only = kl.Reshape(target_shape=list(input_shape[1:]))(input_last_day_only) #remove time axis
    encoded_state = kl.concatenate([encoded_state, input_last_day_only], axis=-1)

    if debug:
        print('ENCODED STATE')
        print(encoded_state.shape.as_list())

    # x = mu.ResStepDecoder(16, 48, 60, (3,3), 2, l2, alpha, name='MyResStepDecoder')(encoded_state)
    x = mu.res_step_decoder_functional(encoded_state, filters=16, upsampled_filters=48, output_steps=output_steps,
                                       kernel_size=(3, 3), depth_multiplier=2, l2_rate=l2, alpha=alpha,
                                       return_sequence=True, anchored=True)

    x = kl.TimeDistributed(kl.Conv2D(48, (1, 1), activation='linear', padding='same', name='nin1',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(32, (1, 1), activation='linear', padding='same', name='nin2',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(16, (1, 1), activation='linear', padding='same', name='nin3',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)

    x = kl.TimeDistributed(kl.Conv2D(8, (1,1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_pre_out')(x)

    x = kl.TimeDistributed(kl.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_out')(x)
    out = x

    return tf.keras.Model(inputs=inputs, outputs=out)


def spatial_feature_pyramid_net_hiddenstate_ND(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Historical input data is encoded using an spatial pyramid network
        https://arxiv.org/pdf/1606.00915.pdf
        https://arxiv.org/pdf/1612.03144.pdf
        https://arxiv.org/pdf/1612.01105.pdf
    Once each input data has been independently encoded though the spatial feature pyramid,
        the historical input data is further encoded into a single feature-cube using a convolutional-LSTM with
        return_sequence=False.
        https://arxiv.org/pdf/1506.04214.pdf
        This tensor is also concatenated with the latest day of historical input data to
        preserve certain input features for which only the most recent day is important.
    The primary difference between this model and spatial_feature_pyramid_net_vectorized_ND is the inclusion
        of a hidden state during the recurrent decoder stage.
        The advantage of this hidden state is to represent factors which may change over time and drive/be correlated
        to the desired process without being directly related to the output.
    From the merged encoded state, a sequence of output steps is produced using the custom ResStepDecoderHS layer
        ResStepDecoder may be described as:
            ResStepDecoderHS(inputEncodedState=E):
                S[0] = G(E)
                HS = Q(E)
                for i in (1, Sequence_Length) do:
                    S[i] := S[i-1] + F(E, S[i-1], HS)
                    HS := HS + V(E, S[i-1], HS)

                return S[1:]

            Where G is a learned function to estimate an initial state from the encoded state E.
            Where F is s learned function to predict the delta between each subsequent time-step.

        ResStepDecoder makes use  of TensorFlow's highly optimized SeparableConv2D to increase computation efficiency
        and reduce the total number of parameters.

    Finally, the sequence of extrapolated states are converted into the ice-presence probability through a
        time-distributed network-in-a-network structure, where the final layer uses a Sigmoid activation function.

    Moderate candidate and good for use with ensembles.
    Best of the non-forecast channel augmented models.

    Matthew King, November 2019 @ NRC
    :return: Keras model object
    """
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    if 'output_steps' in kwargs:
        output_steps = kwargs['output_steps']
        if type(output_steps) != int:
            raise TypeError('Received output_steps of non-int type')
    else:
        output_steps = 30

    if 'leaky_relu_alpha' in kwargs:
        alpha = kwargs['leaky_relu_alpha']
        if type(alpha) != float:
            raise TypeError('Received leaky relu alpha of non-float type')
    else:
        alpha = 0.01
    if 'debug' in kwargs:
        debug = kwargs['debug']
        if type(debug) != bool:
            raise TypeError('Received debug of non-bool type')
    else:
        debug = False

    inputs = tf.keras.Input(shape=input_shape)

    n_features = 24

    full_res_map = mu.spatial_feature_pyramid(inputs, n_features, (3, 3), 8, alpha=alpha, l2_rate=l2,
                                              return_all=False, debug=debug)

    encoded_state = kl.ConvLSTM2D(48-input_shape[-1], (3, 3), padding='same', activation='selu',
                              kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),
                              name='State_Encoder')(full_res_map)

    days_in = input_shape[0]
    input_last_day_only = kl.Cropping3D(cropping=((days_in-1, 0), (0, 0), (0, 0)))(inputs)
    input_last_day_only = kl.Reshape(target_shape=list(input_shape[1:]))(input_last_day_only) #remove time axis
    encoded_state = kl.concatenate([encoded_state, input_last_day_only], axis=-1)

    if debug:
        print('ENCODED STATE')
        print(encoded_state.shape.as_list())

    # x = mu.ResStepDecoder(16, 48, 60, (3,3), 2, l2, alpha, name='MyResStepDecoder')(encoded_state)
    x = mu.res_step_decoder_HS_functional(encoded_state, filters=16, hidden_filters=16, upsampled_filters=48,
                                          output_steps=output_steps,
                                       kernel_size=(3, 3), depth_multiplier=2, l2_rate=l2, alpha=alpha,
                                       return_sequence=True, anchored=True)

    x = kl.TimeDistributed(kl.Conv2D(48, (1, 1), activation='linear', padding='same', name='nin1',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(32, (1, 1), activation='linear', padding='same', name='nin2',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(16, (1, 1), activation='linear', padding='same', name='nin3',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    print(x.shape)
    x = kl.TimeDistributed(kl.Conv2D(8, (1,1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_pre_out')(x)
    print(x.shape)
    x = kl.TimeDistributed(kl.Conv2D(1, (1, 1), activation=None, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_out')(x)
    print(x.shape)
    out = x

    return tf.keras.Model(inputs=inputs, outputs=out)