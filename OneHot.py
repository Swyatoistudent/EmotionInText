import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
train_data = pd.read_csv('train.txt', sep=';', names=['text', 'emotion'])
test_data = pd.read_csv('test.txt', sep=';', names=['text', 'emotion'])
# Train Data Labels
train_data["emotion"] = train_data["emotion"].astype('category')
train_data["emotion_label"] = train_data["emotion"].cat.codes
train_features, train_labels = train_data['text'], tf.one_hot(
 train_data["emotion_label"], 6)
d = dict(enumerate(train_data["emotion"].cat.categories))
print(d)
# Test Data Labels
test_data["emotion"] = test_data["emotion"].astype('category')
test_data["emotion_label"] = test_data["emotion"].cat.codes
test_features, test_labels = test_data['text'], tf.one_hot(
 test_data["emotion_label"], 6)
vocab_size = 15000
vector_size=300
max_seq_len = 20
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(train_data['text'])
sequences_train = tokenizer.texts_to_sequences(train_data['text'])
sequences_test = tokenizer.texts_to_sequences(test_data['text'])
sequences_one = tokenizer.texts_to_sequences(["im feeling really bitter about this one"])
padded_train = pad_sequences(sequences_train, padding='post', maxlen=max_seq_len)
padded_test = pad_sequences(sequences_test, padding='post', maxlen=max_seq_len)
padded_one = pad_sequences(sequences_one,padding='post', maxlen=max_seq_len)