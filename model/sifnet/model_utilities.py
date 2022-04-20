import math
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import LeakyReLU

def spatial_feature_pyramid(input_sequence, full_res_features, kernel_size, max_downsampling_factor, alpha, l2_rate,
                            return_all=False, name_extension="", **kwargs):
    """
    Feature extractor network. Applied independently on a sequence of inputs.


    :param input_sequence: Input tensor. 5D tensor [batch_size, time-steps, Height, Width, Channels]
    :param full_res_features: The number of features to be extracted at full resolution.
                            Each downsampled feature map will have int(base_features/downsampling_factor) features
    :param kernel_size: tuple, kernel size for using in convolutional layers
                        e.g (3,3)
    :param max_downsampling_factor: int. Maximum downsampling factor to be applied. Must be a power of 2 and > 1.
                        e.g 8
    :param alpha: float. Alpha value for use with Leaky ReLU
    :param l2_rate: float. l2 weight regularization rate
    :param return_all: boolean, optional. Default False.
                            True if all resolution feature maps should be returned, otherwise only full resolution.
    :param name_extension: string. Added to each layer's name
    :param kwargs: keyword arguments
                debug: boolean
    :return: Tensor or List of Tensors
    """

    assert(len(input_sequence.shape.as_list()) == 5), 'input_sequence must be a 5D tensor'
    assert(type(full_res_features) == int), 'base_features must be an int'
    assert(type(kernel_size) == tuple and len(kernel_size) == 2), 'kernel_size must be a tuple with len 2'
    assert(type(kernel_size[0]) == int and type(kernel_size[1]) == int), 'Each value of kernel_size must be an int'
    assert(type(alpha) == float), 'alhpa must be a float'
    assert(type(l2_rate) == float), 'l2_rate must be a float'
    assert(type(return_all) == bool), 'return_all must be a bool'
    assert(type(max_downsampling_factor) == int and max_downsampling_factor > 1), 'max_downsampling_factor must be \'' \
                                                                                  'an int greater than 1'
    assert(math.log2(max_downsampling_factor).is_integer()), 'max_downsampling_factor must be a power of 2'
    assert(max_downsampling_factor <= full_res_features), 'max_downsampling_factor must not be greater than base_features'

    if 'debug' in kwargs:
        debug = kwargs['debug']
        assert(type(debug) == bool), 'debug must be a bool'
    else:
        debug = False

    base_features = kl.TimeDistributed(
        kl.Conv2D(full_res_features, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                  kernel_initializer='orthogonal'),
        name='Conv2D_FullSize'+name_extension)(input_sequence)
    base_features = LeakyReLU(alpha)(base_features)

    feature_maps = [base_features]

    exp2 = int(math.log2(max_downsampling_factor))
    downsampling_factors = [2 ** x for x in range(1, exp2+1)]  # [2, 4, 8, ... max_downsampling_factor]

    for downsampling_factor in downsampling_factors:
        # produce a feature map at each downsampling size
        downsampled = kl.TimeDistributed(
            kl.AveragePooling2D((downsampling_factor, downsampling_factor),
                                strides=(downsampling_factor, downsampling_factor),
                                padding='same'),  # padding=same ensures exactly half, quarter, etc. Not actually same.
            name="Downsampling2D_{}".format(downsampling_factor)+name_extension
        )(input_sequence)
        downsampled = kl.TimeDistributed(
            kl.Conv2D(int(full_res_features / downsampling_factor), kernel_size, padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='Conv2d_Downsampled_{}'.format(downsampling_factor)+name_extension
        )(downsampled)
        downsampled = LeakyReLU(alpha)(downsampled)
        feature_maps.append(downsampled)

    if debug:
        print('FEATURE MAPS')
        print([f.shape.as_list() for f in feature_maps])

    updated_feature_maps = [feature_maps[-1]]
    for level in range(len(feature_maps) - 1, 0, -1):  # 3,2,1
        # use the lower resolution feature map to update the feature map at the higher resolution
        f = feature_maps[level]
        n = feature_maps[level - 1]
        features_at_n = int(full_res_features / (2 ** (level - 1)))

        fp = kl.TimeDistributed(
            kl.Conv2DTranspose(features_at_n, (4, 4), (2, 2), padding='same',
                               kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='Pyramid_upflow_{}'.format(level)+name_extension)(f)
        fp = LeakyReLU(alpha)(fp)
        fp_shape = fp.shape.as_list()
        n_shape = n.shape.as_list()
        if debug:
            print("fp_type {}".format(type(fp)))
            print("fp_shape {}".format(fp_shape))
            print("n_shape {}".format(n_shape))
        if not fp_shape == n_shape:
            dh = fp_shape[-2] - n_shape[-2]
            dw = fp_shape[-3] - n_shape[-3]
            fp = kl.Cropping3D(cropping=((0, 0), (0, dw), (0, dh)),
                               name='Upflow_Cropping_{}'.format(level)+name_extension)(fp)
        n = kl.Add()([n, fp])
        feature_maps[level - 1] = n
        updated_feature_maps.append(n)

    if debug:
        print('Updated FEATURE MAPS')
        print([f.shape.as_list() for f in updated_feature_maps])

    if return_all:
        return updated_feature_maps
    else:
        return updated_feature_maps[-1]  # full res only


def res_step_decoder_functional(input_encoded_state, filters, upsampled_filters, output_steps, kernel_size=(3, 3),
                                depth_multiplier=2, l2_rate=1e-4, alpha=3e-2, return_sequence=True, anchored=True,
                                **kwargs):
    """
    A 'layer' which extrapolates a given state across the given number of steps.

    :param input_encoded_state: Input tensor, of shape [batch_size, height, width, channels]
    :param filters: The number of output filters/data channels
    :param upsampled_filters: The number of channels to be used during the update step
    :param output_steps:  The number of timesteps to be processed
    :param kernel_size: kernel size e.g (3,3)s
    :param depth_multiplier: Depth multiplier for SeperableConv layers
    :param l2_rate: l2 weight regularization rate
    :param alpha: alpha for LeakyRelu
    :param return_sequence: True if the whole sequence should be returned, False for only the final state
    :param anchored: True if the update step is computed within context of the input Encoded state, \
                        False for independent update steps
    :param kwargs: keyword arguments
                debug: boolean
    :return:Output tensor
    """

    if 'debug' in kwargs:
        debug = kwargs['debug']
    else:
        debug = False

    assert len(input_encoded_state.shape.as_list()) == 4

    initial_state = kl.SeparableConv2D(filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate))(input_encoded_state)
    initial_state = kl.LeakyReLU(alpha)(initial_state)

    daily_extrapolated_states = []  # placeholder

    upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                          name='Incoming_State_Upsampler')  # NIN

    res_pred = kl.SeparableConv2D(filters, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                                  pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                  name='Residual_Predictor', depth_multiplier=depth_multiplier)

    inner_concatenator = kl.Concatenate(axis=-1, name='inner_concatenator')  # stacks along the feature dimension

    adder = kl.Add(name='residual_adder')

    expand_dims = kl.Lambda(lambda in_tensor: keras.backend.expand_dims(in_tensor, axis=1))

    leakyRelu = kl.LeakyReLU(alpha, name='Leaky_ReLU')

    for i in range(output_steps):
        if i == 0:
            incoming_state = initial_state
        else:
            incoming_state = daily_extrapolated_states[-1]

        upsampled = upsampler(incoming_state)
        upsampled = leakyRelu(upsampled)

        if anchored:
            combined_state = inner_concatenator([upsampled, input_encoded_state])
            residual = res_pred(combined_state)
        else:
            residual = res_pred(upsampled)

        next_state = adder([incoming_state, residual])
        daily_extrapolated_states.append(next_state)

    if debug:
        print('Daily States')
        print(daily_extrapolated_states)

    if return_sequence:
        # Expand dims to add time-step dimension
        for i in range(len(daily_extrapolated_states)):
            s = daily_extrapolated_states[i]
            s = expand_dims(s)
            daily_extrapolated_states[i] = s

        return kl.Concatenate(axis=1)(daily_extrapolated_states)  # concatenate along the time-step dimension
    else:
        return daily_extrapolated_states[-1]

def res_step_decoder_HS_functional(input_encoded_state, filters, hidden_filters, upsampled_filters, output_steps,
                                   kernel_size=(3, 3), depth_multiplier=2, l2_rate=1e-4, alpha=3e-2,
                                   return_sequence=True, anchored=True, **kwargs):
    """
    A 'layer' which extrapolates a given state across the given number of steps. Uses a hidden state for greater
    internal representation power.

    :param input_encoded_state: Input tensor, of shape [batch_size, height, width, channels]
    :param filters: The number of output filters/data channels
    :param hidden_filters: The number of channels in the hidden state
    :param upsampled_filters: The number of channels to be used during the update step
    :param output_steps:  The number of timesteps to be processed
    :param kernel_size: kernel size e.g (3,3)s
    :param depth_multiplier: Depth multiplier for SeperableConv layers
    :param l2_rate: l2 weight regularization rate
    :param alpha: alpha for LeakyRelu
    :param return_sequence: True if the whole sequence should be returned, False for only the final state
    :param anchored: True if the update step is computed within context of the input Encoded state, \
                        False for independent update steps
    :param kwargs: keyword arguments
                debug: boolean
    :return:Output tensor
    """

    if 'debug' in kwargs:
        debug = kwargs['debug']
    else:
        debug = False

    assert len(input_encoded_state.shape.as_list()) == 4

    initial_state = kl.SeparableConv2D(filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                       name='Output_State_Initializer')(input_encoded_state)
    initial_state = kl.LeakyReLU(alpha)(initial_state)

    hidden_state = kl.SeparableConv2D(hidden_filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                       name='Hidden_State_Initializer')(input_encoded_state)
    hidden_state = kl.LeakyReLU(alpha)(hidden_state)

    daily_extrapolated_states = []  # placeholder

    upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                          name='Incoming_State_Upsampler')  # NIN
    hidden_upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(l2_rate),
                                 name='Hidden_State_Upsampler')

    res_pred = kl.SeparableConv2D(filters, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                                  pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                  name='Residual_Predictor', depth_multiplier=depth_multiplier)
    hidden_res_pred = kl.SeparableConv2D(hidden_filters, kernel_size, padding='same',
                                         kernel_regularizer=keras.regularizers.l2(l2_rate),
                                         pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                         name='Hidden_Residual_Predictor', depth_multiplier=depth_multiplier)

    inner_concatenator = kl.Concatenate(axis=-1, name='inner_concatenator')  # stacks along the feature dimension

    adder1 = kl.Add(name='residual_adder')
    adder2 = kl.Add(name='hidden_adder')

    expand_dims = kl.Lambda(lambda in_tensor: keras.backend.expand_dims(in_tensor, axis=1))

    leakyRelu = kl.LeakyReLU(alpha, name='Leaky_ReLU')

    for i in range(output_steps):
        if i == 0:
            incoming_state = initial_state
        else:
            incoming_state = daily_extrapolated_states[-1]

        upsampled = upsampler(incoming_state)
        upsampled = leakyRelu(upsampled)
        hidden_upsampled = hidden_upsampler(hidden_state)

        if anchored:
            combined_state = inner_concatenator([upsampled, input_encoded_state, hidden_upsampled])
            residual = res_pred(combined_state)
            hidden_residual = hidden_res_pred(combined_state)
        else:
            residual = res_pred(upsampled, hidden_upsampled)
            hidden_residual = hidden_res_pred(upsampled, hidden_upsampled)

        next_state = adder1([incoming_state, residual])
        hidden_state = adder2([hidden_state, hidden_residual])

        daily_extrapolated_states.append(next_state)

    if debug:
        print('Daily States')
        print(daily_extrapolated_states)

    if return_sequence:
        # Expand dims to add time-step dimension
        for i in range(len(daily_extrapolated_states)):
            s = daily_extrapolated_states[i]
            s = expand_dims(s)
            daily_extrapolated_states[i] = s

        # concatenate along the time-step dimension
        return kl.Concatenate(axis=1, name='Output_Sequence')(daily_extrapolated_states)
    else:
        return daily_extrapolated_states[-1]

