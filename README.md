# CONVOLUTIONAL NEURAL NETWORK FOR POKÉMON DETECTION
Convolutional neural network experiment for 1st gen Pokémons detection, made with TensorFlow.

## TRY IT OUT!
The model is available on [this HuggingFace repo](https://huggingface.co/Firefly55lm/vera_cnn_v1).
Here is a [Colab Notebook](https://colab.research.google.com/drive/1DX4Yw6NkOcHwxUItwf4LrYGUAl53U97C?usp=sharing) to test it.

## DATASET
The [training dataset](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files) containt 17000 pictures divided in 143 of 1st gen Pokémons.

## ARCHITECTURE
![architecture](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/80dbb7d00a71df9616506fff25a9a9e72badb713/pictures/architecture.png)

The input has been preprocessed with 200x200 resizing and augmented with random orizontal flip and 20% zoom range.

Every convolutional layer has a LeakyReLU (alpha = 0.1) activation function to prevent vanishing gradients and disappearing relu issues, with padding 'same' and 'he_normal' kernel initialization.
In every layers pack there are a Batch Normalization, a Max Pooling layer (2x2) and a 20% Dropout to prevent overfitting.

From the 2nd to the 4th convolutional layers pack, the dilation rate increases from 1x1 to 3x3, to upgrade the area of intervention of every filer
and increase the features detection performance, according with the same principle described in [this paper](https://ieeexplore.ieee.org/document/8756165).
This solution increases the accuracy on validation and test set of 2% (90% accuracy).

After the convolutional layers, the learning and classification is performed by two Dense layers with 512 and 256 weights, with a 40% Dropout.

Here is a plot of the feature maps extracted from every convolutional layers pack:
![feature_maps_plot_1](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/269ee3ed60e0f831b158fb36dc5acd8536b02425/pictures/feature_maps_1.png)
![feature_maps_plot_2](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/269ee3ed60e0f831b158fb36dc5acd8536b02425/pictures/feature_maps_2.png)

### TESTED SOLUTIONS
If you are curious about the visual differences between the feature maps of different kinds of layers, I made a few plots comparing them with the same 6 filters (initializer = GlorotUniform(seed=5)).

This is the list of tested solutions:
- Classic Conv2D, 1 layer, 3x3 kernel
- Separable Depthwise Convolution, 1 layer, 3x3 kernel
- Classic Conv2D, 1 layer, 5x5 kernel
- Dilated Convolution, 1 layer, 3x3 kernel, 2x2 dilation rate
- Dilated Convolution, 1 layer, 3x3 kernel, 3x3 dilation rate
- Dilated Convolution, 3 layers, 3x3 kernel, dilation rates 1x1-2x2-3x3
- Classic Conv2D, 3 layers, 3x3 kernel, 2 layers for MaxPooling 2x2
- Classic Conv2D, 3 layers, 3x3 kernel, 2 layers for AveragePooling 2x2
- Classic Conv2D, 3 layers, 3x3 kernel, 1 layer for MaxPooling 2x2, 1 layer for AveragePooling 2x2
- Dilated Convolution, 3 layers, 3x3 kernel, dilation rates 1x1-2x2-3x3, 2 layers for MaxPooling 2x2
- Dilated Convolution, 3 layers, 3x3 kernel, dilation rates 1x1-2x2-3x3, 2 layers for AveragePooling 2x2
- Dilated Convolution, 3 layers, 3x3 kernel, dilation rates 1x1-2x2-3x3, 1 layer for MaxPooling 2x2, 1 layer for AveragePooling 2x2

![test1](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test1.png)
![test2](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test2.png)
![test3](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test3.png)
![test4](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test4.png)
![test5](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test5.png)
![test6](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test6.png)
![test7](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test7.png)
![test8](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test8.png)
![test9](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test9.png)
![test10](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test10.png)
![test11](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test11.png)
![test12](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/59de42e7b6f02538f0bdff1640ff55c621acc83b/pictures/test12.png)
