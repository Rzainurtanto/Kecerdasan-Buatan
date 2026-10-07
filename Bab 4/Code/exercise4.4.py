# Exercise 4.4 LeNet-5 with Different Number of Filters
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D


def lenet(filters1, filters2):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


# (6, 16) is the original LeNet-5 from Example 4.4
filter_sets = [(6, 16), (3, 8), (12, 32), (32, 64)]
for f1, f2 in filter_sets:
    model = lenet(f1, f2)
    print("Filters (%d, %d) -> Total params: %s" % (f1, f2, format(model.count_params(), ',')))

# Show the summary of one of the modified models
lenet(12, 32).summary()
