# Example 4.17c CNN Visualize Filters.ipynb (Part 3)
from keras.applications.vgg19 import VGG19

model = VGG19()

n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1
