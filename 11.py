# 11 Write a program to develop an RNN model for generating song lyrics and analyse the quality of generated text. CO3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random
# --------------------------------------------------
# 1. Training text
# --------------------------------------------------
text = """
hello from the other side
i must have called a thousand times
to tell you i am sorry
for everything that i have done

we will we will rock you
buddy you are a boy make a big noise
playing in the street gonna be a big man someday

let it be let it be
whisper words of wisdom let it be

here comes the sun
and i say it is all right
""".lower()
text = text.replace("\n", " ")
# 2. Character mappings

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print("Total characters in corpus:", len(text))
print("Unique characters:", len(chars))

# 3. Prepare dataset
seq_length = 40
step = 3

input_sequences = []
target_chars = []

for i in range(0, len(text) - seq_length, step):
    seq = text[i:i + seq_length]
    target = text[i + seq_length]
    input_sequences.append(seq)
    target_chars.append(target)

print("Number of sequences:", len(input_sequences))

X = np.zeros((len(input_sequences), seq_length, len(chars)), dtype=np.float32)
y = np.zeros((len(input_sequences), len(chars)), dtype=np.float32)

for i, seq in enumerate(input_sequences):
    for t, ch in enumerate(seq):
        X[i, t, char_to_idx[ch]] = 1.0
    y[i, char_to_idx[target_chars[i]]] = 1.0

# 4. Build model
model = Sequential([
    LSTM(128, input_shape=(seq_length, len(chars))),
    Dense(len(chars), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

# 5. Train model
model.fit(X, y, batch_size=32, epochs=150, verbose=1)

# 6. Sampling function
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# 7. Generate lyrics
start_index = random.randint(0, len(input_sequences) - 1)
seed_text = input_sequences[start_index]
generated = seed_text

print("\nSeed text:")
print(seed_text)

for _ in range(500):
    x_pred = np.zeros((1, seq_length, len(chars)), dtype=np.float32)
    for t, ch in enumerate(seed_text):
        x_pred[0, t, char_to_idx[ch]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample_with_temperature(preds, temperature=0.8)
    next_char = idx_to_char[next_index]

    generated += next_char
    seed_text = seed_text[1:] + next_char

print("\nGenerated Lyrics:\n")
print(generated)
