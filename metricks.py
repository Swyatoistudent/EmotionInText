import tensorflow.keras as keras
from OneHot import padded_test,padded_train,train_data,test_data,padded_one,d
import numpy as np
import tensorflow as tf

model = keras.models.load_model('models/lstm.hdf5')
y_pred_one_hot_encoded = (model.predict(padded_test)> 0.5).astype("int32")
y_pred_train = np.array(tf.argmax(y_pred_one_hot_encoded, axis=1))
y_pred_one_hot_encoded_one = (model.predict(padded_one))
y_pred_one = np.array(tf.argmax(y_pred_one_hot_encoded_one, axis=1))
print(y_pred_one_hot_encoded_one)
print(d[y_pred_one[0]])
y_pred_one_hot_encoded = (model.predict(padded_train)> 0.5).astype("int32")
y_pred_test = np.array(tf.argmax(y_pred_one_hot_encoded, axis=1))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Test set
print("test")
print(classification_report(test_data['emotion_label'], y_pred_train))

# Training Set
print(classification_report(train_data['emotion_label'], y_pred_test))

# Confusion matrix
cm = confusion_matrix(test_data['emotion_label'], y_pred_train)
print(cm)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(cm, index = d.values(),
columns = d.values())
sn.set(font_scale=1.4)
plt.figure(figsize=(10,10))
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
plt.show()