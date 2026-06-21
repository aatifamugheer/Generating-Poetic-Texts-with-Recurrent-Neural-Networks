import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))

index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

model = tf.keras.models.load_model('textgenerator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

print('---------0.2----------')
print(generate_text(300, 0.2))
print('---------0.4----------')
print(generate_text(300, 0.4))
print('---------0.6----------')
print(generate_text(300, 0.6))
print('---------0.8----------')
print(generate_text(300, 0.8))
print('---------1.0----------')
print(generate_text(300, 1.0))
