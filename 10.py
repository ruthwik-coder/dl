# 10 To develop a GRU-based RNN model for sentiment analysis on the IMDB movie reviews dataset. CO3

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

vocab_size = 10000
max_length = 200
embedding_dim = 32
# Split the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = vocab_size)

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')

x_train = pad_sequences(x_train, maxlen = max_length, padding = 'post', truncating = 'post')
x_test = pad_sequences(x_test, maxlen = max_length, padding = 'post', truncating = 'post')

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')

from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length = max_length),
    GRU(64, return_sequences = True),
    GRU(32),
    Dense(16, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')])

model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_split = 0.2)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

import numpy as np
def predict_sentiment(text):
    #tokenize and pad the text
    text_word_index = imdb.get_word_index()
    tokens = [text_word_index.get(word, 0) for word in text.lower().split()]
    tokens_padded = pad_sequences([tokens], maxlen = max_length, padding ='post', truncating ='post')
    #predict sentiment
    prediction = model.predict(tokens_padded)
    sentiment = 'positive' if prediction >=0.5 else 'negative'
    return sentiment

new_review = "The movie was fantastic and I loved it"
print(f'Sentiment: {predict_sentiment(new_review)}')
new_review = "don't watch this movie worst movie"
print(f'Sentiment: {predict_sentiment(new_review)}')
model.summary()
