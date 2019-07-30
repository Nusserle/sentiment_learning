import random

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import nltk
from elmoformanylangs import Embedder


def read_in(path_to_file):
    out = []
    with open(path_to_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if not row[0] == "2":
                out.append([row[0], row[5]])
    return out


def preprocess(data):
    random.shuffle(data)
    label = []
    tweet = []
    print("Preprocessing...")
    for row in data:
        if row[0] == "0":
            label.append(0)
        else:
            label.append(1)
        cleaned = clean_regex(row[1].lower())
        tokenized = nltk.word_tokenize(cleaned)
        # pos_tagged = nltk.pos_tag(tokenized)
        tweet.append(tokenized)
    print("Done")
    return tweet, label


def clean_regex(tweet):
    cleaned = re.sub('@.*?(?:\s|$)', '', tweet)
    cleaned2 = re.sub('http.*?(?:\s|$)', '', cleaned)
    cleaned3 = re.sub('www.*?(?:\s|$)', '', cleaned2)
    return cleaned3

# 160.000 entries
training_data, training_label = preprocess(read_in("data/training.csv"))
test_data, test_label = preprocess(read_in("data/test.csv"))

x_train = training_data[:140000]
x_val = training_data[140000:]
train_label = np.asarray(training_label[:140000])
val_label = np.asarray(training_label[140000:])


#print("pre")
sents = [["He", "likes", "tea"], ["She", "does" ,"n't", "like" ,"tea"]]
e = Embedder('C:/Users/Tobias.Nusser/PycharmProjects/sentiment_learning/elmo_embedding/144/')
embedding = e.sents2elmo(sents)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics=[metrics.binary_accuracy])
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()