# Exercise 5.11 Federated learning for image classification
# The TensorFlow Federated Colab tutorial needs the tensorflow-federated package, which does
# not support Windows / TensorFlow 2.20. This script implements the same idea, Federated
# Averaging (FedAvg), with plain Keras so it runs locally:
#   1. the server creates a shared model and sends its weights to every client (phone)
#   2. each client trains the model on its own private data for a few local epochs
#   3. clients send back only their weights; the server averages them (weighted by data size)
#   4. repeat for several rounds
# The data never leaves the clients. The result is compared with normal (centralised) training.
import os
os.environ.setdefault("KERAS_HOME", "D:/AI/.keras")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLIENTS = 10
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
SAMPLES_PER_CLIENT = 600

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.0
x_test = x_test[..., None].astype("float32") / 255.0

# Non-IID split, like real phones: every client mostly holds images of 2 digits
rng = np.random.default_rng(0)
idx_by_digit = [np.where(y_train == d)[0] for d in range(10)]
clients = []
for c in range(NUM_CLIENTS):
    digits = [c % 10, (c + 1) % 10]
    main = np.concatenate([rng.choice(idx_by_digit[d], SAMPLES_PER_CLIENT * 2 // 5, replace=False) for d in digits])
    other = rng.choice(len(y_train), SAMPLES_PER_CLIENT // 5, replace=False)
    idx = np.concatenate([main, other])
    clients.append((x_train[idx], y_train[idx]))
    print("client {}: {} images, main digits {}".format(c, len(idx), digits))


def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.SGD(0.05),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Federated Averaging ================================================
server_model = create_model()
client_model = create_model()
fed_acc = []
for r in range(1, ROUNDS + 1):
    global_weights = server_model.get_weights()
    client_weights, client_sizes = [], []
    for x_c, y_c in clients:
        client_model.set_weights(global_weights)  # download the shared model
        client_model.fit(x_c, y_c, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)  # train locally
        client_weights.append(client_model.get_weights())  # upload only the weights
        client_sizes.append(len(x_c))
    # weighted average of every layer's weights
    total = sum(client_sizes)
    new_weights = [sum(w[i] * n / total for w, n in zip(client_weights, client_sizes))
                   for i in range(len(global_weights))]
    server_model.set_weights(new_weights)
    loss, acc = server_model.evaluate(x_test, y_test, verbose=0)
    fed_acc.append(acc)
    print("round {:>2}: global model test accuracy {:.4f}".format(r, acc))

# Centralised baseline: the same data, all uploaded to one server ====
x_all = np.concatenate([c[0] for c in clients])
y_all = np.concatenate([c[1] for c in clients])
central_model = create_model()
history = central_model.fit(x_all, y_all, epochs=ROUNDS * LOCAL_EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(x_test, y_test), verbose=0)
central_acc = history.history["val_accuracy"]
print("\nFederated  final accuracy: {:.4f}".format(fed_acc[-1]))
print("Centralised final accuracy: {:.4f}".format(central_acc[-1]))

plt.plot(range(1, ROUNDS + 1), fed_acc, "o-", label="Federated (FedAvg, {} clients)".format(NUM_CLIENTS))
plt.plot(range(1, ROUNDS + 1), central_acc, "s--", label="Centralised")
plt.xlabel("round / epoch")
plt.ylabel("test accuracy")
plt.title("Exercise 5.11 Federated learning on MNIST")
plt.legend()
plt.grid(True)
plt.savefig("../Output/exercise 5.11.png")
plt.show()
