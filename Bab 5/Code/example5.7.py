# Example 5.7 TransferLearningKeras2.py (Example 5.7a - 5.7e in one file)
# ImageDataGenerator + model.fit (Keras 3 removed fit_generator/evaluate_generator;
# fit/evaluate/predict accept generators directly).
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

# Example 5.7a Load all libraries ====================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16

# Example 5.7b Select image folders and create DataGenerators =======
train_path = os.path.join(os.getcwd(), "Flowers")
test_path = os.path.join(os.getcwd(), "Flowers")

batch_size = 16
image_size = 224  # For Inception use 299

# validation_split=0.3 + subset keeps the last 30% of every class out of training,
# so the "test" images below are never seen during training.
train_datagen = ImageDataGenerator(validation_split=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=vgg16.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    shuffle=True)

# a generator without preprocessing, only to look at the images
preview_generator = ImageDataGenerator(validation_split=0.3).flow_from_directory(
    directory=train_path, target_size=(image_size, image_size), batch_size=batch_size,
    class_mode='categorical', subset='training', shuffle=True)
x_batch, y_batch = next(preview_generator)

it = train_generator.class_indices
print(it.items())
print(train_generator.n)
print(np.unique(train_generator.classes))
num_classes = len(np.unique(train_generator.classes))

fig = plt.figure(figsize=(15, 6))
columns = 4
rows = 4
for i in range(1, columns * rows):
    num = np.random.randint(len(x_batch))
    image = x_batch[num].astype(int)
    fig.add_subplot(rows, columns, i)
    plt.title(str(y_batch[num]))
    plt.imshow(image)
    plt.axis('off')
plt.savefig('../Output/example 5.7b.png')
plt.show()

# Example 5.7c Create, train and save the model ======================
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications import (
    resnet50,
    densenet,
    mobilenet_v2,
    inception_v3,
    inception_resnet_v2,
    efficientnet
)

# Keras Models: https://keras.io/api/applications/
base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = densenet.DenseNet201(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# base_model = efficientnet.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# (when switching the base model, also switch preprocessing_function above)

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential()
model.add(Input(shape=(image_size, image_size, 3)))
# Add the vgg convolutional base model
model.add(base_model)
# Add new layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=5)  # the book trains for 3 epochs

model.save('saved_models/fine_tune.keras')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.savefig('../Output/example 5.7c.png')
plt.show()

# Example 5.7d Predict the test images ===============================
# model = load_model('saved_models/fine_tune.keras')
test_datagen = ImageDataGenerator(validation_split=0.3)  # no preprocessing so images can be shown
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(image_size, image_size),
    color_mode='rgb',
    shuffle=False,
    class_mode='categorical',
    subset='validation',
    batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)
test_generator.reset()

label_map = dict((v, k) for k, v in test_generator.class_indices.items())  # flip k,v
idx = np.random.RandomState(0).choice(nb_samples, columns * rows - 1, replace=False)
fig = plt.figure(figsize=(15, 6))
for i, k in enumerate(idx, start=1):
    x_batch, y_batch = test_generator[k]

    name = model.predict(vgg16.preprocess_input(x_batch.copy()), verbose=0)
    name = np.argmax(name, axis=-1)
    true_name = np.argmax(y_batch, axis=-1)

    predictions = [label_map[n] for n in name]
    true_value = [label_map[n] for n in true_name]

    image = x_batch[0].astype(int)
    fig.add_subplot(rows, columns, i)
    plt.title(str(predictions[0]) + ':' + str(true_value[0]))
    plt.imshow(image)
    plt.axis('off')
plt.savefig('../Output/example 5.7d.png')
plt.show()

# Example 5.7e Predict, save results and evaluate ====================
eval_generator = ImageDataGenerator(validation_split=0.3,
                                    preprocessing_function=vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(image_size, image_size), shuffle=False,
    class_mode='categorical', subset='validation', batch_size=batch_size)

pred = model.predict(eval_generator, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
predictions = [label_map[k] for k in predicted_class_indices]
results = pd.DataFrame({"Filename": eval_generator.filenames,
                        "Predictions": predictions})
results.to_csv("../Output/example 5.7 results.csv", index=False)

score = model.evaluate(eval_generator, verbose=0)
print("Loss: ", score[0], "Accuracy: ", score[1])
