import xarray as xr
import zarr
import os
import pandas as pd
from functools import partial
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import L1L2
from functools import partial

import sys


def split_time(ds, test_size):
    idx = int(len(ds.time) * (1 - test_size))
    train, test = ds.isel(time=slice(None, idx)), ds.isel(time=slice(idx, None))
    return train, test

def fake_loss(y_true, y_pred):
    print(y_pred.shape)
    print(y_true.shape)
    return 0

def apply_tresh(arr, thresh=85, ones_are_open_water=True):
    arr_thresh = np.zeros_like(arr)
    if ones_are_open_water:
        arr_thresh[arr < thresh] = 1
    else:
        arr_thresh[arr > thresh] = 1
    arr_thresh[np.isnan(arr)] = np.nan
    return arr_thresh

def masked_MSE(mask):
    def loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        sq_diff = tf.multiply(tf.math.squared_difference(y_pred, y_true), mask)
        return tf.reduce_mean(sq_diff)
    return loss

def masked_MAE(mask):
    def loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        ab_diff = tf.multiply(tf.math.abs(tf.math.subtract(y_pred, y_true)), mask)
        return tf.reduce_mean(ab_diff)
    return loss

def iiee(y_true, y_pred, thresh=0.85, nan_mask=None):
    # Apply thresh
    y_true = tf.maximum(tf.sign(y_true - thresh), 0)
    y_pred = tf.maximum(tf.sign(y_pred - thresh), 0)
    
    # Apply land mask
    if nan_mask is not None:
        y_true = tf.multiply(y_true, ~nan_mask)
        y_pred = tf.multiply(y_pred, ~nan_mask)
    
    # Calculate IIEE
    diff = tf.subtract(y_true, y_pred)
    o = tf.reduce_sum(tf.math.maximum(diff, 0))
    i = tf.reduce_sum(tf.math.maximum(diff * -1, 0))
    return o + i

loss = masked_MSE

class Climatology:
    def __init__(self):
        pass

    def fit(self, y, dates):
        self.climatologies = xr.DataArray(
            y[:, 0, :, :, 0],
            dims=['time', 'y', 'x'],
            coords={'time': dates}
            ).groupby('time.month').mean()
        
    def predict(self, dates):
        pred_y = xr.DataArray(np.zeros(shape=len(dates)), coords={'time': dates}).groupby('time.month') + self.climatologies
        return np.expand_dims(pred_y.to_numpy(), [1, -1])

class Persistence:
    def __init__(self):
        pass
    
    def predict(self, X):
        """X.shape: (num_features, timesteps, x, y, features) where features[0] is siconc"""
        return X[:, -1:, :, :, 0:1]

    # def predict(self, X):
    #     return np.array([X[0]] + list(X[:-1]))

def is_winter(month):
    return (month >= 1) & (month < 5)

def create_timesteps(arr, num_timesteps=3):
    timesteps = [arr[:-(num_timesteps - 1)]]
    
    for i in range(1, num_timesteps - 1):
        timesteps.append(arr[i:-((num_timesteps-1)-i)])
                
    timesteps.append(arr[(num_timesteps - 1):])
    return np.array(timesteps)

class ModelTester:
    def __init__(self):
        self.data_params = None
        self.model_params = None

        self.num_timesteps = None
        self.binary_sic = None
        self.nan_mask = None
        self.scaler = None

        self.train_Y = None
        self.train_X = None
        self.test_Y = None
        self.test_X = None
        self.dates_train = None
        self.dates_test = None

    def preprocess_data(self,
                        ds,
                        test_size=0.3,
                        deseasonalize=False,
                        only_winter=True,
                        weekly=True,
                        binary_sic=False,
                        only_polynya=True,
                        num_timesteps=3,
                        num_timesteps_predict=1,
                        gap=0,
                        predict_only_sic=True,
                        predict_anomalies=True):
        """
        Creates the training and test datasets and stores them as class attributes.
        :param: ds: xarray DataSet containing input & output variables, with SIC being the first variable
                    (TODO: make this position agnostic)
        """
        
        # Save the input parameters, other than self and the dataset (hacky)
        self.data_params = locals()
        del self.data_params['self']
        del self.data_params['ds']

        self.num_timesteps = num_timesteps
        self.binary_sic = binary_sic
        self.predict_anomalies = predict_anomalies
        
        if weekly:
            ds = ds.resample(time='W').mean()

        # Split by time into train & test
        train, test = split_time(ds, test_size)

        # Include only winter
        if only_winter:
            train = train.sel(time=is_winter(train['time.month']))
            test = test.sel(time=is_winter(test['time.month']))
        
        # Crop to the extent of the polynya
        if only_polynya:
            train = train.isel(x=slice(-32, None), y=slice(None, 32))
            test = test.isel(x=slice(-32, None), y=slice(None, 32))
            
        unscaled_sic_train, unscaled_sic_test = train.ceda_sic.values, test.ceda_sic.values

        # Deseasonalize by removing climatologies (calculated using the entire training dataset)
        if deseasonalize:
            climatologies = train.groupby('time.dayofyear').mean()
            train = train.groupby('time.dayofyear') - climatologies
            test = test.groupby('time.dayofyear') - climatologies

        # Convert into np.array
        train_array, test_array = train.to_array().to_numpy(), test.to_array().to_numpy()

        # Apply (pseudo-)landmask using the NaNs in the first SIC frame
        self.nan_mask = np.isnan(train_array[0][0])

        # Normalization
        self.scaler = StandardScaler()
        train_array = self.scaler.fit_transform(train_array.reshape(-1, np.prod(train_array.shape[1:])).T).T.reshape(train_array.shape)
        test_array = self.scaler.transform(test_array.reshape(-1, np.prod(test_array.shape[1:])).T).T.reshape(test_array.shape)
        
        # Use a 85% threshold to convert to binary sea ice on/off (85 chosen as per extent.ipynb)
        if binary_sic:
            train_array[0] = apply_tresh(unscaled_sic_train, 85).astype(int)
            test_array[0] = apply_tresh(unscaled_sic_test, 85).astype(int)
                
        # Replace NaNs with 0s 
        train_array, test_array = np.nan_to_num(train_array), np.nan_to_num(test_array)
        
        # Create timesteps 
        # This creates (num_timesteps + gap + num_timesteps_predict) timesteps, which is then decomposed
        # Note that this is inefficient and should be revisited (TODO)
        train_X = np.transpose(train_array, axes=[1, 2, 3, 0])
        train_X = create_timesteps(train_X, num_timesteps + gap + num_timesteps_predict)
        train_X = np.transpose(train_X, axes=[1, 0, 2, 3, 4])

        test_X = np.transpose(test_array, axes=[1, 2, 3, 0])
        test_X = create_timesteps(test_X, num_timesteps + gap + num_timesteps_predict)
        test_X = np.transpose(test_X, axes=[1, 0, 2, 3, 4])

        # Split x and y
        train_Y = train_X[:, -(num_timesteps_predict):, :, :, :]
        test_Y = test_X[:, -(num_timesteps_predict):, :, :, :]
        
        # To predict anomalies, we remove the previous timestep (and repeat the first timestep to keep the dimensions the same)
        if predict_anomalies:
            train_Y = train_Y[1:] - train_Y[:-1]
            train_Y = np.array([train_Y[0]] + list(train_Y))
            test_Y = test_Y[1:] - test_Y[:-1]
            test_Y = np.array([test_Y[0]] + list(test_Y))
        
        # Keep only num_timesteps, rather than (num_timesteps + gap + num_timesteps_predict)
        train_X = train_X[:, :num_timesteps, :, :, :]
        test_X = test_X[:, :num_timesteps, :, :, :]

        dates_train = train.time[num_timesteps + gap:]
        dates_test = test.time[num_timesteps + gap:]
        
        # If only predicting SIC, keep only first variable (SIC) but expand to keep the same num. of dimensions
        if predict_only_sic:
            train_Y = np.expand_dims(train_Y[..., 0], -1)
            test_Y = np.expand_dims(test_Y[..., 0], -1)

        # Store data
        self.train_Y = train_Y
        self.train_X = train_X
        self.test_Y = test_Y
        self.test_X = test_X
        self.dates_train = dates_train
        self.dates_test = dates_test

        self.input_shape = train_X.shape[2:]

    def create_model(self,
                     loss,
                     num_convlstm=3,
                     convlstm_filters=[128, 64, 64],
                     convlstm_kernels=[(5, 5), (3, 3), (1, 1)],
                     convlstm_rec_dropout=0.1,
                     convlstm_dropout=0.1,
                     convlstm_kernal_reg=L1L2(0.001, 0.01),
                     num_conv=3,
                     conv_filters=[128, 64, 64],
                     conv_kernels=[(5, 5), (3, 3), (1, 1)],
                     ):
        """
        Creates the model architecture given the inputs & saves the model as a class attribute.
        """
        
        assert num_convlstm == len(convlstm_filters) == len(convlstm_kernels)
        assert num_conv == len(conv_filters) == len(conv_kernels)
        
        # Save model hyperparams, without self and the loss function since it causes issues when pickling (hacky!)
        self.model_params = locals()
        del self.model_params['self']
        del self.model_params['loss']

        # Construct the input layer with no definite frame size.
        inp = layers.Input(shape=(self.num_timesteps, *self.input_shape))

        # Stacked ConvLSTM layers
        x = inp
        for i in range(num_convlstm):
            x = layers.ConvLSTM2D(
                filters=convlstm_filters[i],
                kernel_size=convlstm_kernels[i],
                padding="same",
                return_sequences=True if i < num_convlstm - 1 else False,
                activation="relu",
                dropout=convlstm_dropout,
                recurrent_dropout=convlstm_rec_dropout,
                kernel_regularizer=convlstm_kernal_reg,
            )(x)
            # x = layers.BatchNormalization()(x)

        x = tf.expand_dims(x, 1, name='Reshape')  # Reshape to get the 1 timestep

        # Convolutions to reduce the number of channels
        for i in range(num_conv):
            x = layers.Conv2D(
                filters=conv_filters[i],
                kernel_size=conv_kernels[i],
                padding="same",
                activation="relu",
            )(x)

        # Get latest day
        last_day = layers.Cropping3D(cropping=((self.num_timesteps - 1, 0), (0, 0), (0, 0)))(inp)

        #  Get first variable (siconc)
        last_day = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 5)), data_format='channels_first')(last_day)

        # Concatenate the output from the stacked ConvLSTMs and the last day
        x = layers.concatenate([x, last_day], axis=-1,)

        # Convolutions to reduce the number of channels
        x = layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            # activation="relu",
            padding="same",
            # use_bias=False,
        )(x)

        if self.binary_sic:
            loss = masked_MSE(mask=np.expand_dims(~self.nan_mask, [0, -1]))
        else:
            loss = masked_MSE(mask=np.expand_dims(~self.nan_mask, [0, -1]))

        self.model = keras.models.Model(inp, x)
        self.model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    def train(self, verbose=0, epochs=500, batch_size=20):
        model = self.model
        
        # Define some callbacks to improve training.
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        history = model.fit(
            self.train_X,
            self.train_Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_X, self.test_Y),
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )

        self.history = history.history
        self.model = model

        df = self.get_results()
        return df

    def get_results(self, invert=True):
        """
        Get RMSE and MAE on the test set for the model & two baselines. 
        """

        nan_mask_reshaped = np.expand_dims(~self.nan_mask, [0, -1])
        df = pd.DataFrame(index=['Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE'])

        results = {}
        print(1)
        # NN
        results['NN'] = {}
        results['NN']['Train'] = self.model.predict(self.train_X)
        results['NN']['Test'] = self.model.predict(self.test_X)

        # Persistence
        results['Persistence'] = {}
        results['Persistence']['Train'] = Persistence().predict(self.train_X)
        results['Persistence']['Test'] = Persistence().predict(self.test_X)

        # Climatology
        clim = Climatology()
        clim.fit(self.train_Y, self.dates_train)
        results['Climatology'] = {}
        results['Climatology']['Train'] = clim.predict(self.dates_train)
        results['Climatology']['Test'] = clim.predict(self.dates_test)

        if invert:
            train_Y = self.inverse_sic(self.train_Y)
            test_Y = self.inverse_sic(self.test_Y)
        else:
            train_Y = self.train_Y
            test_Y = self.test_Y

        for model_name, predictions in results.items():
            if invert:
                predictions['Train'] = self.inverse_sic(predictions['Train'], anomalies=model_name=='Persistence')
                predictions['Test'] = self.inverse_sic(predictions['Test'], anomalies=model_name=='Persistence')
            df[model_name] = [
                masked_MAE(nan_mask_reshaped)(train_Y, predictions['Train']).numpy(),
                np.sqrt(masked_MSE(nan_mask_reshaped)(train_Y, predictions['Train']).numpy()),
                masked_MAE(nan_mask_reshaped)(test_Y, predictions['Test']).numpy(),
                np.sqrt(masked_MSE(nan_mask_reshaped)(test_Y, predictions['Test']).numpy()),
            ]

        self.df = df
        return df

    def inverse_sic(self, arr, anomalies=True):
        """
        Invert the SIC transformations to get SIC values
        """
        if self.predict_anomalies & anomalies:
            arr = arr[1:] + arr[:-1]
            arr = np.array([arr[0]] + list(arr))
        expanded_arr = np.concatenate([arr, np.zeros(shape=(*arr.shape[:-1], self.scaler.n_features_in_ - 1))], axis=-1).transpose([4, 1, 2, 3, 0])
        inversed = self.scaler.inverse_transform(expanded_arr.reshape(-1, np.prod(expanded_arr.shape[1:])).T).T.reshape(expanded_arr.shape)

        return inversed.transpose([4, 1, 2, 3, 0])[..., 0:1]

    def save(self, model_name, save_dir):
        """
        Save object as pickle and model using the keras save() function since pickle cannot handle the keras model.
        """
        filename = save_dir + model_name

        # Save model separately 
        if self.model is not None:
            self.model.save(filename + '_model.p')

        # Duplicate yourself and do ridiculous gymnastics because Keras cannot handle my custom loss
        model = self.model
        self.model = None
        obj = copy.deepcopy(self)
        self.model = model

        # Delete model from object since it cannot be serialized 
        obj.model = None

        # Remove data too since it's too big...
        obj.train_Y = None
        obj.train_X = None
        obj.test_Y = None
        obj.test_X = None

        # Save without model
        with open(filename + '.p', 'wb') as f:
            pickle.dump(obj, f)

    def load(self, model_name, save_dir, load_model=False):
        """
        Load a previously saved ModelTester object.
        """
        filename = save_dir + model_name

        # Load object without model
        with open(filename + '.p', 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

        # Load model and add as attribute to self
        if load_model:
            self.model = keras.models.load_model(filename + '_model.p',
                                                custom_objects={'loss': loss})
    
    def plot_examples(self, num_frames, i_start=0, set_='test', diff_from=None):
        """
        Plot example predictions for the model and baseline models.
        """
        X = self.train_X if set_ == 'train' else self.test_X
        y = self.train_Y if set_ == 'train' else self.test_Y

        dates = self.dates_train if set_ == 'train' else self.dates_test

        clim = Climatology()
        clim.fit(self.train_Y, self.dates_train)
        
        frame_dates = dates[i_start:i_start + num_frames]
        
        # Add a frame because the first one gets deleted
        truth = y[i_start:i_start + num_frames + 1]
        pred_nn = self.model.predict(X[i_start:i_start + num_frames + 1])
        pred_pers = Persistence().predict(X[i_start:i_start + num_frames + 1])
        pred_clim = clim.predict(dates[i_start:i_start + num_frames + 1])
        
        truth = self.inverse_sic(truth)[1:]
        pred_nn = self.inverse_sic(pred_nn)[1:]
        pred_pers = self.inverse_sic(pred_pers, anomalies=False)[1:]
        pred_clim = self.inverse_sic(pred_clim)[1:]
        
        plt.imshow(np.ma.masked_where(self.nan_mask, truth[0, 0, :, :, 0]))
        plt.colorbar()
        
        cmap = 'viridis'
        
        if diff_from is not None:
            cmap = 'RdBu'
            if diff_from == 'Persistence':
                truth = truth - pred_pers
                pred_nn = pred_nn - pred_pers
            elif diff_from == 'Climatology':
                truth = truth - pred_clim
                pred_nn = pred_nn - pred_clim
            else:
                raise ValueError('Spell it right, Doofus.')

        fig, axs = plt.subplots(num_frames, 4, figsize=(10, num_frames * 2.5))

        for j in range(num_frames):
            dt = str(frame_dates[j].values)[:10]
            
            axs[j][0].imshow(np.ma.masked_where(self.nan_mask, truth[j, 0, :, :, 0]), vmin=0, vmax=100, cmap=cmap)
            axs[j][1].imshow(np.ma.masked_where(self.nan_mask, pred_nn[j, 0, :, :, 0]), vmin=0, vmax=100, cmap=cmap)
            axs[j][2].imshow(np.ma.masked_where(self.nan_mask, pred_pers[j, 0, :, :, 0]), vmin=0, vmax=100, cmap='viridis')
            axs[j][3].imshow(np.ma.masked_where(self.nan_mask, pred_clim[j, 0, :, :, 0]), vmin=0, vmax=100, cmap='viridis')
            
            if j == 0:
                axs[j][0].set_title(f'Ground truth')
                axs[j][1].set_title(f'Model')
                axs[j][2].set_title(f'Persistence')
                axs[j][3].set_title(f'Climatology')
            
            axs[j][0].set_ylabel(dt)
                
        plt.tight_layout()
        return fig, axs