# Exercise 4.3 MLP with 5 Neurons vs 10 Neurons in the Dense Layer
import time
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=7)

features = X_train.shape[1]


def build_and_train(neurons):
    model = Sequential()
    model.add(Input(shape=(features,)))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

    start = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=0)
    train_time = time.time() - start

    results = model.evaluate(X_test, y_test, verbose=0)
    return train_time, results[0], results[1], model.count_params()


results = {}
for neurons in [10, 5]:
    results[neurons] = build_and_train(neurons)

print("\nNeurons | Params | Training time (s) | Loss (MSE) | Accuracy")
for neurons, (t, loss, acc, params) in results.items():
    print("%7d | %6d | %17.2f | %10.4f | %.4f" % (neurons, params, t, loss, acc))
