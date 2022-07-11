import os
import json
import time

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.applications.vgg19 import preprocess_input

from .datasets import read_TFRecords
from .models.CNN.super_resolution import srgan
from .utils import display_lr_hr_sr

class Trainer:
    """
    Training of Deep Learning models.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    model_name: string
        Name of the model
    model: Model
        Keras model
    scaling_factor: int
        Scaling Factor for Super-Resolution.
    loss: function
        Loss method to evaluate how well the model fits the given data. We support all the loos functions from https://keras.io/losses/
    metrics: list of strings
        Name of a function that is used to judge the performance of your model. We support all the metric functions from https://keras.io/metrics/
    learning_rate: float
        A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
    """
    def __init__(self, folder_path, dataset_name, model_name, model, scaling_factor, loss, metrics, learning_rate):
        self.folder_path = folder_path
        self.dataset_name = dataset_name
        self.model = model
        self.loss = loss
        self.metrics = metrics
        with open(os.path.join(self.folder_path, self.dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)

        self.model_dir = os.path.join(self.folder_path, self.dataset_name, model_name)

        # Add model parameters
        self.params["model_name"] = model_name
        self.params["learning_rate"] = learning_rate
        self.params["scaling_factor"] = scaling_factor
        self.params["model_dir"] = self.model_dir

        # Create directory to store the model
        if not os.path.isdir(self.model_dir ):
            os.mkdir(self.model_dir )

    def train(self, normalize_rgb=True, norm_range = [0,1], batch_size=32, shuffle_size=2000, epochs=25):
        """
        Train the Model.
        ----------
        normalize_rgb: boolean
            Boolean to normalize RGB bands.
        batch_size: int
            A number of samples processed before the model is updated. 
            The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
        shuffle_size: int
            Number of samples to be shuffled.
        epochs: int
            Number of complete passes through the training dataset.
        """ 
        # Add training parameters
        self.params["normalize_rgb"] = normalize_rgb
        self.params["norm_range"] = norm_range
        self.params["batch_size"] = batch_size
        self.params["shuffle_size"] = shuffle_size
        self.params["epochs"] = epochs

        # Compile Keras model
        optimizer = Adam(learning_rate=self.params["learning_rate"])
        self.model.compile(optimizer, loss=self.loss, metrics=self.metrics)

        # Pass a tfrecord
        TFRecord = read_TFRecords(self.folder_path, self.dataset_name, normalize_rgb, norm_range, batch_size, shuffle_size, self.params['scaling_factor'])
        self.training_dataset = TFRecord.get_training_dataset()
        self.validation_dataset = TFRecord.get_validation_dataset()

        # Fit model
        self.model.fit(
            x=self.training_dataset,
            steps_per_epoch=int(self.params['training_size'] / batch_size),
            epochs=epochs,
            validation_data=self.validation_dataset,
            validation_steps=int(self.params['validation_size'] / batch_size),
            verbose=1)

        # Save the model weights and metrics
        self.model.save_weights(os.path.join(self.model_dir, 'model_weights.h5'))

        self.metrics = pd.DataFrame(self.model.history.history)
        with open(os.path.join(self.model_dir, 'training_history.json'), mode='w') as f:
            self.metrics.to_json(f)

        # Save the parameters in a json file.
        with open(os.path.join(self.model_dir, "training_params.json"), 'w') as f:
            json.dump(self.params, f)

class regressionTrainer(Trainer):
    """
    Training of SRGAN's generator model.
    ----------
    """
    def __init__(self, folder_path, dataset_name, model_name, model, scaling_factor=None, learning_rate=1e-4):
        super().__init__(folder_path, dataset_name, model_name, model, scaling_factor, loss=MeanSquaredError(), metrics=['mse'], learning_rate=learning_rate)

    def train(self, normalize_rgb=True, norm_range = [-1,1], batch_size=32, shuffle_size=2000, epochs=25):
        super().train(normalize_rgb, norm_range, batch_size, shuffle_size, epochs)

class segmentationTrainer(Trainer):
    """
    Training of SRGAN's generator model.
    ----------
    """
    def __init__(self, folder_path, dataset_name, model_name, model, scaling_factor=None, learning_rate=1e-4):
        super().__init__(folder_path, dataset_name, model_name, model, scaling_factor, loss=CategoricalCrossentropy(), metrics=['accuracy'], learning_rate=learning_rate)

    def train(self, normalize_rgb=True, norm_range = [0,1], batch_size=32, shuffle_size=2000, epochs=25):
        super().train(normalize_rgb, norm_range, batch_size, shuffle_size, epochs)

class SrganTrainer:
    """
    Training of SRGAN model.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    model_name: string
        Name of the model
    generator: Model
        GAN's generator model.
    discriminator: Model
        GAN's discriminator model.
    scaling_factor: int
        Scaling Factor for Super-Resolution.
    learning_rate: float
        A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
    """
    def __init__(self, folder_path, dataset_name, model_name, generator, discriminator, content_loss='VGG22', learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):
        self.model_dir = os.path.join(folder_path, dataset_name, model_name)
        with open(os.path.join(folder_path, dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)

        # Add model parameters
        self.params["model_name"] = model_name
        self.params["content_loss"] = content_loss
        if type(learning_rate) == float:
            self.params["learning_rate"] = learning_rate
        else:
            self.params["learning_rate"] = tf.keras.optimizers.schedules.serialize(learning_rate)
        self.params["model_dir"] = self.model_dir


        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

        # Create directory to store the model
        if not os.path.isdir(self.model_dir ):
            os.mkdir(self.model_dir )

        # Save checkpoints
        checkpoint_dir = 'training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.model_dir, checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

    def train(self, dataset, epochs, steps):
        """
        Train the model.
        ----------
        dataset: TFRecordDataset 
            TFRecord dataset .
        epochs: int
            Number of complete passes through the training dataset.
        steps: int
            Number of batches in the training dataset.
        """ 
        perceptual_loss = []
        discrimination_loss = []
        for epoch in range(1, epochs+1):
            print ('-'*15, 'Epoch %d' % epoch, '-'*15)
            start = time.time()
            
            pls_metric = Mean()
            dls_metric = Mean()
            step = 0

            for lr, hr in tqdm(dataset.take(steps)):
                step += 1

                pl, dl = self.train_step(lr, hr)
                pls_metric(pl)
                dls_metric(dl)

                #if step % 1 == 0:
                #    #print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                #    perceptual_loss.append(pls_metric.result().numpy())
                #    discrimination_loss.append(dls_metric.result().numpy())
                #    pls_metric.reset_states()
                #    dls_metric.reset_states()

            # Save the checkpoints every 5 epochs
            if (epoch) % 5 == 0:
                print('Saving the checkpoints ')
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        
            print(f'Epoch {epoch}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
            perceptual_loss.append(pls_metric.result().numpy())
            discrimination_loss.append(dls_metric.result().numpy())
            pls_metric.reset_states()
            dls_metric.reset_states()

        # Save the model's weights and metrics
        print('Saving the model')
        self.generator.save_weights(os.path.join(self.model_dir, 'generator_weights.h5'))
        self.discriminator.save_weights(os.path.join(self.model_dir, 'discriminator_weights.h5'))

        self.metrics = pd.DataFrame({'perceptual_loss': perceptual_loss, 'discrimination_loss': discrimination_loss})
        with open(os.path.join(self.model_dir, 'training_history.json'), mode='w') as f:
            self.metrics.to_json(f)

        # Save the parameters in a json file.
        with open(os.path.join(self.model_dir, "training_params.json"), 'w') as f:
            json.dump(self.params, f)

        # Display last batch
        display_lr_hr_sr(self.generator, lr, hr)

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self.content_loss(hr, sr)
            gen_loss = self.generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self.discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss


