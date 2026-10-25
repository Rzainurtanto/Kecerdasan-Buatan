# Example 5.8 TransferLearningKeras3.py
# A small CNN built layer by layer and trained from scratch (no pre-trained weights).
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

train_path = os.path.join(os.getcwd(), "Flowers")
batch_size = 16
image_size = 224

train_datagen = ImageDataGenerator(validation_split=0.3, rescale=1. / 255,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    directory=train_path, target_size=(image_size, image_size), batch_size=batch_size,
    class_mode='categorical', subset='training', shuffle=True)
test_generator = ImageDataGenerator(validation_split=0.3, rescale=1. / 255).flow_from_directory(
    directory=train_path, target_size=(image_size, image_size), batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=False)
num_classes = len(np.unique(train_generator.classes))

# Create and train a simple self defined model ======================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()
model.add(Input(shape=(image_size, image_size, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))  # the book uses sigmoid; softmax is the match for categorical_crossentropy
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10)  # the book trains for 1 epoch

model.save('saved_models/fine_tune_cnn.keras')

score = model.evaluate(test_generator, verbose=0)
print("Loss: ", score[0], "Accuracy: ", score[1])

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.savefig('../Output/example 5.8.png')
plt.show()
