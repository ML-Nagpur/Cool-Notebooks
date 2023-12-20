import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""# Load data of IMDB movie reviews"""

(train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=10000)

"""# Pad sequences to have consistent length"""

max_sequence_length = 200
X_train = pad_sequences(train_data, maxlen=max_sequence_length)
X_test = pad_sequences(test_data, maxlen=max_sequence_length)

"""# Build the RNN model"""

model = models.Sequential()
model.add(layers.Embedding(input_dim=10000, output_dim=16, input_length=max_sequence_length))
model.add(layers.SimpleRNN(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

"""# Train the model

"""

model.fit(X_train, train_labels, epochs=3, batch_size=64, validation_split=0.2)

"""# Evaluate the model on the test set"""

loss, accuracy = model.evaluate(X_test, test_labels)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save model
model.save('imdb_rnn_model.h5')