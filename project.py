import numpy as np
from tensorflow.keras.datasets import imdb

max_features = 1000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training data: {len(X_train)}, Labels: {len(y_train)}")
print(f"Testing data: {len(X_test)}, Labels: {len(y_test)}")

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from tensorflow.keras.preprocessing import sequence
max_len = 200

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
