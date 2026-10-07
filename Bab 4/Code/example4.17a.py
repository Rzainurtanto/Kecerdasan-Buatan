# Example 4.17a CNN Visualize Filters.ipynb (Part 1)
# Modified from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

# load vgg model
from keras.applications.vgg19 import VGG19

# load the model
model = VGG19()
# summarize the model
model.summary()
