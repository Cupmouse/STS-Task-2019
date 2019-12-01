from gensim.models import FastText
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.model_selection import KFold
import sklearn.metrics.pairwise as skpw
import numpy as np
import keras

# This takes a little bit time
w2v = FastText.load_fasttext_format('drive/My Drive/cc.en.300')
# w2v = KeyedVectors.load_word2vec_format('drive/My Drive/GoogleNews-vectors-negative300.bin', binary=True)

def load_x(filename):
  with open(filename, 'rt') as file:
    x = [ [ sent[:-1].split(' ') for sent in line.split('\t') ] for line in file.readlines() ]
    return [ pair[0] for pair in x ], [ pair[1] for pair in x ]
  
def to_tensor(docs):
  max_words = max(*[ len(doc) for doc in docs])

  x = np.zeros(shape=(len(docs), max_words, 300))

  for i, doc in enumerate(docs):
    for j, word in enumerate(doc):
      if word in w2v.wv:
        x[i][j] = w2v.wv[word]
      else:
        print('Word excluded: %s' % word)
  return x


x_train_l, x_train_r = load_x('drive/My Drive/STS.input.preprocessed.train.txt')
x_train = [ to_tensor(x_train_l), to_tensor(x_train_r) ]
y_train = np.loadtxt('drive/My Drive/STS.gs.preprocessed.train.txt')
y_train /= 5

x_l, x_r = load_x('drive/My Drive/STS.input.preprocessed.images.txt')
x_test = [ to_tensor(x_l), to_tensor(x_r) ]
y_test = np.loadtxt('drive/My Drive/STS.gs.preprocessed.test.txt')
y_test /= 5

len(x_train_l)

def train(x_train, y_train, x_test):
  model = keras.models.Sequential()
  l_in = keras.layers.Input(shape=(None, 300), name='input_left')
  r_in = keras.layers.Input(shape=(None, 300), name='input_right')
  # This layer will mask vectors in tensor where padding was applied so it will
  # be ignored
  masking = keras.layers.Masking(mask_value=0., name='input_masking')
  l_masking = masking(l_in)
  r_masking = masking(r_in)
  # These both RNN networks have the same weight (same network)
  rnn = keras.layers.LSTM(100, activation='tanh', name='rnn_forwards')
  l_rnn = rnn(l_masking)
  r_rnn = rnn(r_masking)
  # brnn = keras.layers.LSTM(100, activation='tanh', go_backwards=True, name='rnn_backwards')
  # l_brnn = brnn(l_masking)
  # r_brnn = brnn(r_masking)
  # This is the same as consine simirality.
  # concat = keras.layers.Concatenate(name='concatenation')
  # l_concat = concat([l_rnn, l_brnn])
  # r_concat = concat([r_rnn, r_brnn])
  # add = keras.layers.Add(name='Addition')
  # l_added = add([l_rnn, l_brnn])
  # r_added = add([r_rnn, r_brnn])
  # dense = keras.layers.Dense(100)
  # l_dense = dense(l_added)
  # r_dense = dense(r_added)
  cos = keras.layers.Dot(normalize=True, axes=1, name='cosine_similarity')([l_rnn, r_rnn])
  model = keras.models.Model([l_in, r_in], cos)
  model.compile(optimizer=keras.optimizers.RMSprop(5e-5), loss='binary_crossentropy', metrics=['mae'])
  model.summary()

  # Train the network
  h = model.fit(x_train, y_train, batch_size=75, epochs=15, validation_split=0.2)
  model.save('drive/My Drive/model_out')
  # Predict both train and test input
  a_train = model.predict(x_train)
  a_test = model.predict(x_test)

  hist_mae = h.history['mean_absolute_error']
  hist_valmae = h.history['val_mean_absolute_error']
  epochs = np.arange(len(hist_mae))

  plt.plot(epochs, hist_valmae)
  plt.plot(epochs, hist_mae)
  plt.show()

  return np.reshape(a_train, (-1,)), np.reshape(a_test, (-1,))

# Train all, test
p_train, p_test = train(x_train, y_train, x_test)

print(ss.pearsonr(p_train, y_train))
print(ss.pearsonr(p_test, y_test))

plt.scatter(y_train, p_train)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()
plt.scatter(y_test, p_test)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

# KFold Tests
# kfold = KFold(n_splits=10, shuffle=True)

# kf_train = np.zeros(shape=(10,))
# kf_test = np.zeros(shape=(10,))
# for i, (train_index, test_index) in enumerate(kfold.split(x_train[0])):
#   print(i)
#   kfold_x_train = [ x_train[0][train_index], x_train[1][train_index] ]
#   kfold_y_train = y_train[train_index]
#   kfold_x_test = [ x_train[0][test_index], x_train[1][test_index] ]
#   p_train, p_test = train(kfold_x_train, kfold_y_train, kfold_x_test)
#   print(p_train)
#   kf_train[i] = ss.pearsonr(p_train, kfold_y_train)[0]
#   kf_test[i] = ss.pearsonr(p_test, kfold_y_train)[0]
# print('kfold train ave')
# print(np.mean(kf_train))
# print('kfold test ave')
# print(np.mean(kf_test))