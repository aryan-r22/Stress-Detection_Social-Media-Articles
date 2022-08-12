from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
from sklearn.utils import class_weight
import numpy
import random

import pandas as pd
import re
import ast
import emoji

random.seed(23)

path_to_file = input("Enter file path")
df=pd.read_csv(path_to_file,encoding = 'utf-8')

net_features=df['text'].tolist()
net_labels=df['label'].tolist()

temp = list(zip(net_features, net_labels)) 
random.shuffle(temp) 
net_features, net_labels = zip(*temp)

training_size=round(len(net_features)*0.6)
test_val_sz = round((len(net_features)-training_size)/2)
training_sentences = list(net_features[0:training_size])
validation_sentences = list(net_features[training_size:training_size+test_val_sz])
training_labels = list(net_labels[0:training_size])
validation_labels = list(net_labels[training_size:training_size+test_val_sz])

test_sentences= list(net_features[training_size+test_val_sz:])
test_labels = list(net_labels[training_size+test_val_sz:])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(training_sentences,
                            truncation=True,
                            padding=True)
val_encodings = tokenizer(validation_sentences,
                            truncation=True,
                            padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    training_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    validation_labels
))

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

weights = class_weight.compute_class_weight('balanced',
                                            numpy.unique(training_labels),
                                            training_labels)
a={0:weights[0],1:weights[1]}

callback=tf.keras.callbacks.EarlyStopping(patience=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
h=model.fit(train_dataset.shuffle(100).batch(8),
          epochs=20,
          batch_size=8,
          validation_data=val_dataset.shuffle(100).batch(8),
          callbacks=[callback],class_weight=a)

feat = test_sentences
label = test_labels

predicted_labels=[]
counter=0
for y in feat:
  predict_input = tokenizer.encode(y,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")
  tf_output = model.predict(predict_input)[0]
  tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()[0]
  predicted_labels.append(int(tf.argmax(tf_prediction)))
  counter+=1

m1=tf.keras.metrics.Recall()
m2=tf.keras.metrics.Precision()
m3 = tf.keras.metrics.Accuracy()
m1.update_state(label,predicted_labels)
m2.update_state(label,predicted_labels)
m3.update_state(label,predicted_labels)
recall=m1.result().numpy()
precision=m2.result().numpy()
acc = m3.result().numpy()
print(f"Recall = {recall}")
print(f"Precision = {precision}")
f1=2*recall*precision/(recall+precision)
print(f"F1 = {f1}")
print(f"Accuracy = {acc}")

model.save_pretrained("/content/drive/MyDrive/Stress Detection_NTU/Models/DistilBERT_Reddit/BT_Summ_Pegasus")










