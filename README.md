# Variational Autoencoder
This is the code for implementation of some cnn architactures using PyTorch, demonstrated over Imagenette dataset (https://github.com/fastai/imagenette)
## Usage
Run the following command

```python main.py```

### Parameters

``--model-name`` - model to run (default: googlenet)

  So far available models are: alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, wideresnet,   
  densenet121, densenet169, densenet201, densenet201, densenet264, googlenet, xception 

```--batch-size``` - input batch size for training (default: 32)

```--epochs``` - number of epochs to train (default: 100)

```--lr``` - learning rate step size (default: 10)

```--seed``` - random seed (default: 1)

```--no-cuda``` - disables CUDA training

```--save-mode``` - for saving the current model

### Example

```python main.py --model-name=xception --batch-size=64 --epochs=200 --save-model```

To see more about the training process, check log files in the logs directory

