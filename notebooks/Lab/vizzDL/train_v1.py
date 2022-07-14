import os
import json
from unicodedata import name

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from . import model
from .datasets import read_TFRecords

class Trainer(object):
    """
    Training of Deep Learning models.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    """
    def __init__(self, folder_path, dataset_name):
        self.folder_path = folder_path
        self.dataset_name = dataset_name
        with open(os.path.join(self.folder_path, self.dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)

        
    def create_model(self, model_type='CNN', model_output='segmentation', model_architecture='segnet', scaling_factor=None,\
                                model_name = None, model_description='', learning_rate=1E-3, output_activation='', loss=None,  metrics=None):
        """
        Create the Keras Model.
        ----------
        model_type: string
            Type of neural network. We support: 
                - Convolutional Neural Network (CNN)
                - multilayer perceptron (MLP)
                - super resolution (SR)
        model_output: string
            Output of the neural network. We support:
                - regression
                - segmentation
                - super_resolution
        model_architecture: string
            Name of the architecture to be used (e.g.: segnet, deepvel, unet ...)
        scaling_factor: int
            Scaling Factor for Super-Resolution.
        model_name: string
            Name of the model
        model_description: string
            Description of the model
        learning_rate: float
            A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
        output_activation: string
            Name of the last activation function. We support all the activations from https://keras.io/activations/
        loss: string
            Name of a method to evaluate how well the model fits the given data. We support all the loos functions from https://keras.io/losses/
        metrics: list of strings
            Name of a function that is used to judge the performance of your model. We support all the metric functions from https://keras.io/metrics/
        """
        self.model_types = ['CNN', 'MLP'] 
        self.CNN_outputs = ['regression', 'segmentation', 'super_resolution']  
        self.CNN_architectures = ['deepvel','segnet','unet', 'enhance', 'edsr', 'srgan', 'wdsr']
        self.MLP_outputs = ['regression']  
        self.MLP_architectures = ['sequential1']

        self.model_structure = {'model_type': {
            'CNN': {'model_output': {
                'regression': {'model_architecture': [
                        'deepvel','segnet','unet']},
                'segmentation': {'model_architecture': [
                        'deepvel','segnet','unet']},
                'super_resolution': {'model_architecture': [
                                        'enhance', 'edsr', 'srgan', 'wdsr']}}
            },
            'MLP': {'model_output': {
                'regression': {'model_architecture': [
                        'sequential1']}}}}}

        error_dic = {
            'CNN': {
                'outputs': self.CNN_outputs,
                'architectures': self.CNN_architectures,
                'kernel_size': (self.params.get('kernel_size') > 1),
                'kernel_error': 'kernel_size > 1'
            },
            'MLP': {
                'outputs': self.MLP_outputs,
                'architectures': self.MLP_architectures,
                'kernel_size': (self.params.get('kernel_size') == 1),
                'kernel_error': 'kernel_size = 1'
            }
        }

        if model_type in self.model_types:
            if (model_output in error_dic[model_type]['outputs']) and (model_architecture in error_dic[model_type]['architectures']):
                self.model_output = model_output
                self.model_architecture = model_architecture
                if error_dic[model_type]['kernel_size']:
                    self.model_type = model_type
                else:
                    m = error_dic[model_type]['kernel_error']
                    raise ValueError(f'Model type {model_type} only supported when {m}. Current kernel_size is equal to {str(self.kernel_size)}')
            else:
                raise ValueError(f'Unsupported model structure. Check compatibilities: {json.dumps(self.model_structure)}')
        else:
            raise ValueError(f'Unsupported model type. Choose between [CNN, MLP]')

        # Add model parameters
        self.params["model_name"] = model_name
        self.params["model_description"] = model_description
        self.params["model_type"] = self.model_type
        self.params["model_output"] = self.model_output
        self.params["model_architecture"] = self.model_architecture
        self.params["learning_rate"] = learning_rate
        if scaling_factor:
            self.params["scaling_factor"] = scaling_factor
        if output_activation:
            self.params["output_activation"] = output_activation

        if not loss:
            if self.model_output == 'regression':
                self.loss = 'mse'
            if self.model_output == 'segmentation':
                self.loss = 'categorical_crossentropy'
            if self.model_output == 'super_resolution':
                self.loss = 'mse'
        if not metrics:
            if self.model_output == 'regression':
                self.metrics = ['mse']
            if self.model_output == 'segmentation':
                self.metrics = ['accuracy']
            if self.model_output == 'super_resolution':
                self.metrics = ['mse']

        # Select the model
        selected_model = model.select_model(self.params)  

        input_shape = (None, None, len(self.params['in_bands']))
        if self.model_output == 'super_resolution':
                self.keras_model = selected_model(inputShape = input_shape, nClasses = len(self.params['out_bands']), scale = self.params['scaling_factor'])
        else:
                self.keras_model = selected_model(inputShape = input_shape, nClasses = len(self.params['out_bands']))

        # Print model structure
        #self.keras_model.summary()

    def train(self, normalize_rgb=True, batch_size=32, shuffle_size=2000, epochs=25):
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
        self.params["batch_size"] = batch_size
        self.params["epochs"] = epochs
        self.params["shuffle_size"] = shuffle_size

        # Compile Keras model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"])
        self.keras_model.compile(optimizer, loss=self.loss, metrics=self.metrics)

        # Pass a tfrecord
        TFRecord = read_TFRecords(self.folder_path, self.dataset_name, normalize_rgb, [0,1], batch_size, shuffle_size, self.params['scaling_factor'])
        self.training_dataset = TFRecord.get_training_dataset()
        self.validation_dataset = TFRecord.get_validation_dataset()

        # Create directory to store the model
        if not os.path.isdir(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"])):
            os.mkdir(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"]))

        # Setup TensorBoard callback.
        #tensorboard_cb = tf.keras.callbacks.TensorBoard(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"], 'logs'), histogram_freq=1)

        # Setup EarlyStopping callback.
        early_stop = EarlyStopping(monitor='val_loss',patience=3)
        
        self.keras_model.fit(
            x=self.training_dataset,
            steps_per_epoch=int(self.params['training_size'] / batch_size),
            epochs=epochs,
            validation_data=self.validation_dataset,
            validation_steps=int(self.params['validation_size'] / batch_size),
            verbose=1)#,
            #callbacks=[early_stop])

        # Save the model
        #tf.keras.models.save_model(self.keras_model, os.path.join(self.folder_path, self.dataset_name, self.params["model_name"], 'model'), save_format="tf")

        # Save the model weights and metrics
        self.keras_model.save_weights(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"], 'model_weights.h5'))

        self.metrics = pd.DataFrame(self.keras_model.history.history)
        with open(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"], 'training_history.json'), mode='w') as f:
            self.metrics.to_json(f)

        # Save the parameters in a json file.
        with open(os.path.join(self.folder_path, self.dataset_name, self.params["model_name"], "training_params.json"), 'w') as f:
            json.dump(self.params, f)

    

