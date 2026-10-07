# Exercise 4.7 MNIST CNN with Different Number of Filters and Filter Size
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def train_cnn(filters, kernel_size):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    train_time = time.time() - start

    scores = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), train_time, 100 - scores[1] * 100


# (32, (5, 5)) is the original model from Example 4.9
configs = [(32, (5, 5)), (64, (5, 5)), (32, (3, 3))]
results = []
for filters, kernel_size in configs:
    print("\nTraining Conv2D(%d, %s) ..." % (filters, kernel_size))
    results.append((filters, kernel_size) + train_cnn(filters, kernel_size))

print("\nFilters | Kernel | Params    | Training time (s) | CNN Error")
for filters, kernel_size, params, t, error in results:
    print("%7d | %-6s | %9s | %17.2f | %.2f%%" % (filters, "%dx%d" % kernel_size, format(params, ','), t, error))
