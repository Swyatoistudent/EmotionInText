import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import visualkeras
from ann_visualizer.visualize import ann_viz;


train_data = pd.read_csv('train.txt', sep=';', names=['text', 'emotion'])
test_data = pd.read_csv('test.txt', sep=';', names=['text', 'emotion'])

# Train Data Labels
train_data["emotion"] = train_data["emotion"].astype('category')
train_data["emotion_label"] = train_data["emotion"].cat.codes
train_features, train_labels = train_data['text'], tf.one_hot(train_data["emotion_label"], 6)

# Test Data Labels
test_data["emotion"] = test_data["emotion"].astype('category')
test_data["emotion_label"] = test_data["emotion"].cat.codes
test_features, test_labels = test_data['text'], tf.one_hot(test_data["emotion_label"], 6)

# Keras imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LSTM
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
vocab_size = 15000
vector_size= 300
max_seq_len = 20
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(train_data['text'])
sequences_train = tokenizer.texts_to_sequences(train_data['text'])
sequences_test = tokenizer.texts_to_sequences(test_data['text'])
padded_train = pad_sequences(sequences_train, padding='post', maxlen=max_seq_len)
padded_test = pad_sequences(sequences_test, padding='post', maxlen=max_seq_len)
def lstm_model():
    model = Sequential()
    model.add(
    Embedding(input_dim=vocab_size,
    output_dim=vector_size,
    input_length=max_seq_len))
    model.add(Dropout(0.6))
    model.add(LSTM(max_seq_len, return_sequences=True))
    model.add(LSTM(6))
    model.add(Dense(6, activation='softmax'))
    return model

callbacks = [
 keras.callbacks.EarlyStopping(monitor="val_loss",
 patience=4,
 verbose=1,
 mode="min",
 restore_best_weights=True),
 keras.callbacks.ModelCheckpoint(filepath='models/lstm.1.hdf5',
 verbose=1,
 save_best_only=True)
]
model = lstm_model()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf.config.run_functions_eagerly(True)
history = model.fit(padded_train,
 train_labels,
 validation_split=0.33,
 callbacks=callbacks,
 epochs=20)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()