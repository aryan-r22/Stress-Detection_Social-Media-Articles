from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
from sklearn.utils import class_weight
import numpy
import random


path_to_file = input("Enter file path")
df=pd.read_csv(path_to_file,encoding = 'utf-8')

random.seed(23)

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
path_to_plm = input("Enter path to pretrained model")
model = TFDistilBertForSequenceClassification.from_pretrained(path_to_plm, num_labels=2)

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

'''
## Reddit Title
Recall = 0.9742646813392639
Precision = 0.9888059496879578
F1 = 0.9814814589478188
Accuracy = 0.9819982051849365

## Twitter Full
Recall = 0.9295929670333862
Precision = 0.8399602174758911
F1 = 0.8825065192600638
Accuracy = 0.8735954761505127

## Twitter NAdvt
Recall = 0.8779527544975281
Precision = 0.9065040946006775
F1 = 0.8920000135841367
Accuracy = 0.8682926893234253

'''