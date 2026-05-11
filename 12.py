# 12 Write a program to build a conversational chatbot using a Bidirectional LSTM model and assess its response generation performance CO3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
#Step 1: define a simple dataset
conversations = [
    ("Hi", "Hello!"),
    ("How are you?", "I'm good, thank you!"),
    ("What's your name?", "I'm a chatbot."),
    ("What do you do?", "I talk to people."),
    ("Bye", "Goodbye!")
]

#split input and output pairs
inputs, outputs = zip(*conversations)
#step 2: CREATE TOKENIZER AND FIT on both input and output
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + outputs)
vocab_size = len(tokenizer.word_index) +1
# convert texts to sequences
input_seqs = tokenizer.texts_to_sequences(inputs)
output_seqs = tokenizer.texts_to_sequences(outputs)
#pad sequences to equal Length
max_len = max(max(len(seq) for seq in input_seqs), max(len(seq) for seq in output_seqs))
X= pad_sequences(input_seqs , maxlen=max_len, padding = 'post')
y =pad_sequences(output_seqs , maxlen=max_len, padding = 'post')
#convert output to one-hot
y = tf.keras.utils.to_categorical(y, num_classes = vocab_size)
#Step 3: define the bidirectional lstm model
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length= max_len))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(Dense(vocab_size, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Step 4: Train the model
model.fit(X, y, epochs = 200, verbose = 0)
def generate_response(model, tokenizer, user_input, max_len):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen = max_len, padding = 'post')
    pred = model.predict(padded)
    pred_seq = np.argmax(pred, axis = -1)[0]
    words = [tokenizer.index_word.get(i, '') for i in pred_seq]
    response = ' '.join(word for word in words if word != '')
    return response.strip()
#test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    response = generate_response(model, tokenizer, user_input, max_len)
    print("Bot:", response)
