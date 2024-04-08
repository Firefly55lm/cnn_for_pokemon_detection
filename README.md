# CONVOLUTIONAL NEURAL NETWORK FOR POKÉMON DETECTION

NB: Work in progress

### TRY IT OUT!
Here is a [Colab Notebook](https://colab.research.google.com/drive/1DX4Yw6NkOcHwxUItwf4LrYGUAl53U97C?usp=sharing) to test the model

### ARCHITECTURE
![architecture](https://github.com/Firefly55lm/cnn_for_pokemon_detection/blob/067771a06bc5dae4ff873c48fcdff20d04ac8e58/architecture.png)

The input has been preprocessed with 200x200 resizing and augmented with random orizontal flip and 20% zoom range.

Every convolutional layer has a LeakyReLU (alpha = 0.1) activation function to prevent vanishing gradients and disappearing relu issues, with padding 'same' and 'he_normal'kernel initialization.
In every layers pack there are a Batch Normalization, a Max Pooling layer (2x2) and a 20% Dropout to prevent overfitting.

From the 2nd to the 4th convolutional layers pack, the dilation rate increases from 1x1 to 3x3, to upgrade the area of intervention of every filer
and increase the features detection performance, according to the same principle described in [this paper](https://ieeexplore.ieee.org/document/8756165).
This solution increased the accuracy on validation and test set of 2%.

After the convolutional layers, the learning and classification is performed by two Dense layers with 512 and 256 weights, with a 40% Dropout.
