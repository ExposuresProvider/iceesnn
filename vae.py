from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from plotnine import *
import itertools

def load_data(input_file, columns):
    df0 = pd.read_csv(input_file)
    cols = columns if columns is not None else df0.columns
    df = df0[cols]
    onehotencodeddf = pd.get_dummies(df, columns=cols)
    return onehotencodeddf


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def from_dummies(data, categorical_cols, prefix_sep='_'):
    out = data.copy()

    for col_parent in categorical_cols:
        
        filter_col = [col for col in data if col.startswith(col_parent + prefix_sep)]
        cols_with_ones = np.argmax(data[filter_col].values, axis=1)
        cols = data[filter_col].columns
        org_col_values = []
        for row, col in enumerate(cols_with_ones):
            org_col_values.append(cols[col][len(col_parent+prefix_sep):])
                    
        out[col_parent] = pd.Series(org_col_values).values
        out.drop(filter_col, axis=1, inplace=True)    
        
    return out


def sample_decoder(decoder,
                   data,
                   one_hot_columns,
                   categorical_columns,
                   model_name,
                   latent_dim,
                   n=30):
    grid_x = np.linspace(-4, 4, n)

    z_sample = np.array([list(element) for element in itertools.product(*([grid_x] * latent_dim))])
    x_decoded = decoder.predict(z_sample)
    df0 = pd.DataFrame(data=x_decoded, columns=one_hot_columns)

    df = from_dummies(df0, categorical_columns)
    print(df)
    filename = f"{model_name}_samples.csv"
    df.to_csv(filename, index=False)

    if len(categorical_columns) >= 3:
        df2 = df.groupby(list(df.columns)).size().reset_index(name="Frequency")
        cols = list(df2.columns)
        (ggplot(df2, aes(x = cols[1], y = "np.log(Frequency + 1)", color = cols[2])) + geom_point() + geom_line() + facet_grid(f"{cols[0]} ~ {cols[2]}")).save(f"{model_name}_samples_plot.png")
    



# MNIST dataset
def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def get_model(original_dim, scale_width, latent_dim, loss_function="xent"):
    # network parameters
    input_shape = (original_dim, )

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    intermediate_dim = original_dim * scale_width // 2 
    dims = []
    while intermediate_dim >= latent_dim * 2:
        x = Dense(intermediate_dim, activation='relu')(x)
        dims.append(intermediate_dim)
        intermediate_dim //= 2
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file=f'{model_name}_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    for intermediate_dim in reversed(dims):
        x = Dense(intermediate_dim, activation='relu')(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file=f'{model_name}_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    # VAE loss = mse_loss or xent_loss + kl_loss
    if loss_function == "mse":
        reconstruction_loss = mse(inputs, outputs)
    elif loss_function == "xent":
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
    else:
        raise RuntimeError(f"unsupported loss function {loss_function}")

    reconstruction_loss *= original_dim

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",
                        help="Load h5 model trained weights")
    parser.add_argument("--input_file",
                        help="Load h5 model trained weights",
                        type=str,
                        required=True)
    parser.add_argument("--columns",
                        help="Load h5 model trained weights",
                        type=str,
                        nargs="+")
    parser.add_argument("--loss_function",
                        help="loss function: mse | xent",
                        type=str,
                        default="xent")
    parser.add_argument("--epochs",
                        help="number of epochs",
                        type=int,
                        default=50)
    parser.add_argument("--latent_dim",
                        help="dimension of latent space",
                        type=int,
                        default=2)
    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=128)
    parser.add_argument("--width_scale",
                        help="width scale",
                        type=int,
                        default=2)
    parser.add_argument("--model_name",
                        help="prefix for file names",
                        type=str,
                        default="vae_icees")
    parser.add_argument("-n",
                        help="number of samples to generate",
                        type=int,
                        default=128)
    args = parser.parse_args()

    input_file = args.input_file
    columns = args.columns
    model_name = args.model_name

    df = load_data(input_file, columns)
    print(df)
    print(df.columns)
    data = df.values

    x_train, x_test = train_test_split(data)
    
    original_dim = x_train.shape[1]
    latent_dim = args.latent_dim

    vae, encoder, decoder = get_model(original_dim, args.width_scale, latent_dim, args.loss_function)

    vae.summary()
    plot_model(vae,
               to_file=f'{model_name}.png',
               show_shapes=True)


    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(x_test, None))
        vae.save_weights(f'{model_name}.h5')

    sample_decoder(decoder,
                   data,
                   one_hot_columns = df.columns,
                   categorical_columns = columns,
                   latent_dim = latent_dim,
                   model_name=model_name,
                   n=args.n)
