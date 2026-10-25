# Example 5.6 TransferLearningKeras.py (Example 5.6a - 5.6h in one file)
# The book uses its own "SkinArea" dataset (6 skin-area classes), which is not public.
# Here the same code runs on "Flowers": 5 classes x 40 images, one sub-folder per class.
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

# Example 5.6a Load all the libraries ================================
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, to_categorical, load_img

# Example 5.6b Select the image folder ===============================
datapath = os.path.join(os.getcwd(), "Flowers")

# grab the image paths and randomly shuffle them
imagePaths = sorted(os.path.join(r, f) for r, _, fs in os.walk(datapath) for f in fs
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
random.seed(42)
random.shuffle(imagePaths)

# initialize the number of epochs to train for, initial learning rate,
# and batch size
batch_size = 32
epochs = 10  # the book uses 50; 10 is enough here and much faster on a CPU
lr = 1e-3  # Initial Learning rate

image_size = 224  # For Inception use 299
num_classes = 0  # number of classes for classification

# initialize the data and labels
data = []
labels = []
classNum = []
classNames = []

# Get a list of subfolders as labels
subfolders = sorted(f.path for f in os.scandir(datapath) if f.is_dir())
i = 0
for folder in subfolders:
    s = os.path.basename(folder)
    classNum.append(i)
    classNames.append(s)
    i = i + 1

num_classes = len(classNum)
print(classNum)
print(classNames)
print(num_classes)

# Example 5.6c Create training data and labels =======================
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(classNum[classNames.index(label)])

print(labels[0:20])

# Example 5.6d Split the data ========================================
from tensorflow.keras.applications import vgg16

# The book divides by 255; VGG16's own preprocess_input matches how its ImageNet weights were trained
data = vgg16.preprocess_input(np.array(data, dtype="float32"))
labels = np.array(labels)
print(labels[0:30])

# Split the data into training (75%) and testing (25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42, stratify=labels)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=len(classNum))
testY = to_categorical(testY, num_classes=len(classNum))

# Example 5.6e Create and train the model ============================
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    resnet50,
    densenet,
    mobilenet_v2,
    inception_v3,
    efficientnet
)

# Keras Models: https://keras.io/api/applications/
base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = densenet.DenseNet201(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = efficientnet.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential()
model.add(Input(shape=(image_size, image_size, 3)))
# Add the convolutional base model
model.add(base_model)
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compile the model (a smaller learning rate than Adam's default keeps the big Flatten head stable)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Start the training process
history = model.fit(trainX, trainY, validation_split=0.30, batch_size=batch_size,
                    epochs=epochs, verbose=2)
# Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

# save the model
model.save('saved_models/model.keras')

# Example 5.6f Evaluate ==============================================
model = load_model('saved_models/model.keras')
score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Example 5.6g Predictions on the test data ==========================
names = model.predict(testX, verbose=0)
print(len(testX))
print(trainX[0, :].shape)
for p, t in zip(names.argmax(axis=1), testY.argmax(axis=1)):
    print("predicted: {:<11} true: {}".format(classNames[p], classNames[t]))

# Example 5.6h Prediction on a particular image file =================
testfile = os.path.join(datapath, 'roses', sorted(os.listdir(os.path.join(datapath, 'roses')))[0])
test_image = load_img(testfile, target_size=(image_size, image_size))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = vgg16.preprocess_input(test_image)  # same preprocessing as the training data
results = model.predict(test_image, verbose=0)
print(results)
prediction = "Unknown"
for i, name in enumerate(classNames):
    if results[0][i] > 0.4:
        prediction = name
        break
print(prediction)

plt.figure()
plt.title(os.path.basename(testfile) + " -> " + prediction)
plt.imshow(load_img(testfile))
plt.savefig('../Output/example 5.6.png')
plt.show()
