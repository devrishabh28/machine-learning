import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love Machine Learning',
    'Elden Ring is my favourite game!',
    'I love to play games'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
print(word_index)
print(sequences)
print(padded)

test_data = [
    'i want to be super rich!',
    'i am a god'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
