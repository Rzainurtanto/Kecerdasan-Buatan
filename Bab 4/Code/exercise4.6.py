# Exercise 4.6 Add Another Conv2D, MaxPooling and Dropout Layer
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Original model from Example 4.7 ================================
model1 = Sequential()
model1.add(Input(shape=(28, 28, 1)))
model1.add(Conv2D(32, (5, 5), activation='relu'))
model1.add(MaxPooling2D())
model1.add(Dropout(0.2))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

# Modified model with another Conv2D, MaxPooling, Dropout ========
model2 = Sequential()
model2.add(Input(shape=(28, 28, 1)))
model2.add(Conv2D(32, (5, 5), activation='relu'))
model2.add(MaxPooling2D())
model2.add(Dropout(0.2))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D())
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())

print("Original model params :", format(model1.count_params(), ','))
print("Modified model params :", format(model2.count_params(), ','))
