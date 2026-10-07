# Exercise 4.5 AlexNet with Different Number of Dense Layer Units
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def alexnet(units1, units2, units3):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(filters=96, kernel_size=(11, 11), activation='relu', strides=(4, 4), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=256, kernel_size=(11, 11), activation='relu', strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Fully Connected layers with different number of units
    model.add(Flatten())
    model.add(Dense(units1, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units3, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
    return model


# (4096, 4096, 1000) is the original AlexNet from Example 4.5
unit_sets = [(4096, 4096, 1000), (2048, 2048, 500), (1024, 1024, 256), (512, 256, 128)]
for u1, u2, u3 in unit_sets:
    model = alexnet(u1, u2, u3)
    print("Dense units (%d, %d, %d) -> Total params: %s" % (u1, u2, u3, format(model.count_params(), ',')))

# Show the summary of one of the modified models
alexnet(1024, 1024, 256).summary()
