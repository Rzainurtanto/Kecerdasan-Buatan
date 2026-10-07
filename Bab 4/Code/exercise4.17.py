# Exercise 4.17 Visualize the Feature Maps of Other Convolutional Layers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims


def plot_feature_maps(feature_maps, title):
    # plot the first 64 feature maps
    col = 8
    row = min(int(feature_maps.shape[3] / col), 8)
    ix = 1
    plt.figure(figsize=(12, 12))
    plt.suptitle(title)
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


# load the image with the required shape
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

vgg = VGG19()
# Select the hidden layers to visualize:
# 2 = block1_conv2, 4 = block2_conv1, 7 = block3_conv1, 12 = block4_conv1
for n in [2, 4, 7, 12]:
    # redefine model to output right after the hidden layer
    model = Model(inputs=vgg.inputs, outputs=vgg.layers[n].output)
    feature_maps = model.predict(img, verbose=0)
    print("Layer", n, vgg.layers[n].name, "-> Feature maps: ", feature_maps.shape)
    plot_feature_maps(feature_maps, "Layer %d (%s)" % (n, vgg.layers[n].name))
