# Deep learning robotic guidance for autonomous vascular access, [nature machine intelligence](https://www.nature.com/natmachintell/)

The below code is designed to help users to evaluate the deep learning models described in the paper for segmentation, classification, localization, and tracking of near infrared (NIR) and duplex ultrasound (DUS) image sequences. Using the code requires basic knowledge of Python programming, Tensorflow, and training deep neural networks to understand the training and evaluation procedures.

## Dependencies

The models were developed using Python3 with Tensorflow 1.4. The Python dependencies for running the code include

```
Tensorflow 1.4
SimpleITK
Matplotlib
ImageIO
Numpy
```

## Data preparation

Example test data are provided in the folders ```/data/nir_test``` and ```/data/dus_test```. Each directory contains multiple sets of contiguous example test sequences. Individual data are in the form of .mat files, structured as follows:

```
/data/nir_test/sequenceX/data_YYYYYY.mat
  Image_left - WxH rectified left NIR stereo image input.
  Image_right - WxH rectified right NIR stereo image input.
  Mask_left - WxH optional binary input mask of the left arm segmentation.
  Mask_right - WxH optional binary input mask of the right arm segmentation.
  Label_left - WxH left binary segmentation label.
  Label_right - WxH right binary segmentation label.
  Disparity - WxHx1x2 stereo disparity map label.
  
/data/dus_test/sequenceX/data_YYYYYY.mat
  Image_bmode - WxH rectified left NIR stereo image input.
  Image_doppler - WxHx3 rectified right NIR stereo image input.
  Label_mask - WxH binary segmentation label.
  Label_info - Information about the vessel labels, including the vessel class and name.
```

Output predictions may be (optionally) written to disk as .png image files.

## Train and test models from scratch

The Python scripts for training the models from scratch are included under ```/nir_trainer``` and ```/dus_trainer```. Both follow a similar set up:

```nir_create_lists.py``` and ```dus_create_lists.py``` parse the data into training, validation, and test splits. The script use a list structure given by

```
datalist = [[dataPath1, startingFrameIndex, endingFrameIndex, timeSteps], 
            [dataPath2, startingFrameIndex, endingFrameIndex, timeSteps],
            ...
           ]
```

For single frame model (FCN), data txt files should be a list of image file names separated by line breaks. For time series (Rec-FCN), data txt files should be a list of image file names separated by line breaks, which each image sequence is separated by a dash ("-"), i.e.,

```
relative_path\sequence1\data_000001.mat
relative_path\sequence1\data_000002.mat
relative_path\sequence1\data_000003.mat
-
relative_path\sequence2\data_000001.mat
relative_path\sequence2\data_000002.mat
relative_path\sequence2\data_000003.mat
-
relative_path\sequence3\data_000001.mat
relative_path\sequence3\data_000002.mat
relative_path\sequence3\data_000003.mat
```

Every N video sequences should have the same length, where N is equal to the desired training batch size. Separate data lists can be create in this manner to handle training, validation, and testing. Example data lists can be found under ```/nir_trainer/datalists``` and ```/dus_trainer/datalists```.

```nir_run.py``` and ```dus_run.py``` are the entry points for running the models. These scripts set the relevant paths needed for model training and testing and allow the user to set input parameters that define the model structure and training strategy. For example,

```nir_trainer.py``` and ```dus_trainer.py``` are the main classes that set up the model structure according to the user inputs, build the Tensorflow model graph, create the batches, perform (online) data augmentation, and run the training/test sessions. 

The main function definitions within these classes for model training and testing are provided below (along with default input values):

```
class ModelTrainer:
    def train(self, training_list,
                    validation_list,
                    testing_list,
                    model_type = 'RFCN',
                    model_checkpoint = None, 
                    restartdirectoryname = None,
                    restart_from = -1,
                    epochs = 101,
                    epochs_save = 5,
                    learning_rate = 0.00001,
                    loss_type='WCE+Dice+MSE',
                    loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
                    pos_weight = 3.0,
                    threshold = 0.5,
                    k_composite=(0.5, 0.5, 0.5, 0.5, 0.5),
                    min_max=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                    n_augs = 1,
                    batch_size = 8,
                    batch_size_val = 8,
                    window_size = 1,
                    input_channels = 2,
                    output_channels = 2,
                    n_channels = 4,
                    crop_from_seg_center = False,
                    apply_augmentation = False,
                    shuffle_sequence_order = True,
                    apply_reverse_order = True,
                    predict_from_label = False,
                    write_prediction_text_file = False,
                    write_images = False,
                    show_images = False,
                    show_activation_maps = False
                    ):
                    
    def test(self, testing_list,
                   model_type = 'RFCN',
                   model_checkpoint = None, 
                   restartdirectoryname = None,
                   restart_from = -1,
                   loss_type = 'WCE',
                   loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
                   pos_weight = 3.0,
                   threshold = 0.5,
                   k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
                   min_max = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                   n_augs = 1,
                   batch_size = 8,
                   window_size = 1,
                   input_channels = 2,
                   output_channels = 2,
                   n_channels = 4,
                   crop_from_seg_center = False,
                   apply_augmentation = False,
                   predict_from_label = False,
                   write_prediction_text_file = False,
                   write_images = False,
                   show_images = False,
                   show_activation_maps = False
                   ):
                   
Other class functions for training and testing batches:
    def createMiniBatch() # creates mini-batches for training
    def createMiniBatchForTesting() # creates mini-batches for testing
    def computeCompositeTransforms() # creates the composite transforms for data augmentation using SimpleITK
    def getResults() # handles data writing, data display, and loss calculation during training and testing
```

The model definitions are found in ```fcn_model/fcn_model.py``` and includes the following network layer definitions:

```
def prelu() # parametric leaky Relu nonlinear activation function

def convolution_2d() # standard convolution operator
def deconvolution_2d() # standard transpose convolution operator
def convolution_block() # convolution block (3x3 kernel size) (used in encoder)
def convolution_block_2() # convolution block (3x3 kernel size) preceded by skip connection (used in decoder)
def convolution_block_in() # convolution block (variable kernel size) (used in encoder)
def convolution_block_2_in() # convolution block (variable kernel size) preceded by skip connection (used in decoder)
def convolution_block_in_batchnorm() # convolution block (variable kernel size) with batch norm layers (used in encoder)
def convolution_block_2_in_batchnorm() # convolution block (variable kernel size) with batch norm layers and preceded by skip connection (used in decoder)

def down_convolution() # stride-2 convolutions for down sampling (used in encoder)
def up_convolution() # stride-2 tranpose convolutions for up sampling (used in decoder)
def up_convolution_resize() # stride-2 resize-up convolutions for up sampling (used in decoder)

def fcn_encoder() # encoder
def fcn_encoder_batchnorm() # encoder with batch normalization
def fcn_decoder() # decoder
def fcn_decoder_batchnorm() # decoder with batch normalization
def fcn_encoder_decoder() # encoder-decoder

def convGRU_2d_gate() # convolutional gated recurrent unit
def convGRU_2d_output() # convolutional gated recurrent unit output
def fcn_convGRU() # convolutional GRU structure

def fcn_outer() # overall encoder-decoder structure
def fcn_outer_batchnorm() # overall encoder-decoder structure with batch normalization
def rfcn_outer() # overall recurrent encoder-decoder structure
def rfcn_outer_batchnorm() # overall recurrent encoder-decoder structure with batch normalization

```

Loss functions used in the reported studies are defined in ```/fcn_model/fcn_lossfunc.py```:

```
def sigmoid()
def computeModifiedHausdorffDistance2D_TF()
def computeModifiedHausdorffDistance2D()
def computeWeightedCrossEntropyWithLogits_TF()
def computeWeightedCrossEntropyWithLogits()
def computeDiceLoss_TF()
def computeGeneralizedDiceLoss2D_TF()
def computeGeneralizedDiceLoss_TF()
def computeDiceScorePrediction()
def computeMeanSquaredLoss_TF()
def computeMeanSquaredLoss()
def setLoss_TF()
def evalLoss_TF()
def evalLoss_customized()

```

Finally, a number of utility functions are provided in ```/fcn_model/fcn_utilities.py```:

```
def showActivationMaps() # show network activation maps at each spatial resolution layer
def showImagesAsSubplots() # show input and output images as subplots
def writeOutputsToFile() # write network predictions to specified file path
```

## Trainable model checkpoints

Example trainable (not frozen) Tensorflow models may be found in ```/nir_trainer/checkpoints/``` and ```dus_trainer/checkpoints/```.

## Citation
Please cite our paper if you use the data or code
```
@article{
  title={Deep learning robotic guidance for autonomous vascular access},
  author={Chen, Alvin and Balter, Max and Maguire, Timothy and Yarmush, Martin L},
  journal={Nature Machine Intelligence},
  volume={X},
  number={X},
  pages={X},
  year={2020},
  publisher={Nature Publishing Group}
}
```
