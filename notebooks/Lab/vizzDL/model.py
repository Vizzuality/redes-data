from .models.CNN.regression import deepvel as CNNregDeepVel, segnet as  CNNregSegNet, unet as CNNregUNet
from .models.CNN.segmentation import deepvel as CNNsegDeepVel, segnet as  CNNsegSegNet, unet as CNNsegUNet
from .models.CNN.super_resolution import enhance as CNNsrEnhance, edsr as CNNspEDSR, srgan as CNNspSRGAN, wdsr as CNNspWDSR
from .models.MLP.regression import sequential1 as MLPregSequential1

def select_model(params):
    """
    Select model.
    ----------
    params: dict
        Dictionary with the TFRecords creation and training parameters.
    """
    # Model's dictionary
    models = {'CNN':
              {
                  'regression': 
                  {
                      'deepvel': CNNregDeepVel.create_keras_model,
                      'segnet': CNNregSegNet.create_keras_model,
                      'unet': CNNregUNet.create_keras_model,
                  },
                  'segmentation': 
                  {
                      'deepvel': CNNsegDeepVel.create_keras_model,
                      'segnet': CNNsegSegNet.create_keras_model,
                      'unet': CNNsegUNet.create_keras_model,
                  },
                  'super_resolution': 
                  {
                      'enhance': CNNsrEnhance.create_keras_model,
                      'edsr': CNNspEDSR.create_keras_model,
                      'srgan': CNNspSRGAN.create_keras_model,
                      'wdsr': CNNspWDSR.create_keras_model,
                  }
              }, 
              'MLP': 
              {
                  'regression': 
                  {
                      'sequential1': MLPregSequential1.create_keras_model,
                  }
              }
             }
    
    return models.get(params.get('model_type')).get(params.get('model_output')).get(params.get('model_architecture'))