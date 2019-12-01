from gensim.models import FastText
import numpy as np
import sklearn.metrics.pairwise as skpw
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scipy.stats as ss
import keras.models as models
import keras.layers as layers
import keras.losses as losses
import keras.optimizers as optim

model = FastText.load_fasttext_format('drive/My Drive/cc.en.300')

vec1 = model.wv['machine']
vec2 = model.wv['computer']
vec3 = model.wv['oji90390ejdf0d']
np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
vec3

CHAR_REMOVE_TARGETS = list(',.()')
with open('drive/My Drive/stop_words.txt', 'rt') as file:
	STOP_WORDS = [ line[:-1] for line in file.readlines() ]
def preprocess(sent_array):
	# Convert all words to lower case
	sent_array = map(lambda word: word.lower(), sent_array)
	# Remove a specific character if it exists
	sent_array = map(lambda word: ''.join([ ch for ch in word if ch not in CHAR_REMOVE_TARGETS ]), sent_array)
	# Remove stop-words
	sent_array = filter(lambda word: word not in STOP_WORDS, sent_array)
	return list(sent_array)

TRAIN_FILES = [
               'MSRpar',
               'MSRvid',
               'OnWN',
               'SMTeuroparl',
               'deft-forum',
               'deft-news',
               'headlines',
              #  'tweet-news'
               ]
x_train = []
y_train = np.array([])
for file in TRAIN_FILES:
  with open('drive/My Drive/data/STS.input.%s.txt' % file) as fin:
    x_train += fin.readlines()
  y_train = np.concatenate((y_train, np.loadtxt('drive/My Drive/data/STS.gs.%s.txt' % file)))
x_train = [ [ preprocess(sent[:-1].split(' ')) for sent in line.split('\t') ] for line in x_train ]

with open('drive/My Drive/data/STS.input.images.txt') as fin:
  x_test = fin.readlines()
x_test = [ [ preprocess(sent[:-1].split(' ')) for sent in line.split('\t') ] for line in x_test ]
y_test = np.loadtxt('drive/My Drive/data/STS.gs.images.txt')

v_train = np.zeros(shape=(len(x_train), 2, 300), dtype=np.float64)
v_test = np.zeros(shape=(len(x_test), 2, 300), dtype=np.float64)

def document2vector(doc, v):
  for i, pair in enumerate(doc):
    for j, sent in enumerate(pair):
      v[i][j] = np.sum([ model.wv[word] for word in sent if word in model.wv ], axis=0)

document2vector(x_train, v_train)
document2vector(x_test, v_test)


kfold = KFold(n_splits=10, shuffle=True)

def csim(v, a):
  for i, pair in enumerate(v):
    a[i] = skpw.cosine_similarity(pair)[0][1] * 5

def train(x_train, x_test):
  a_train = np.zeros(shape=(len(x_train),), dtype=np.float64)
  a_test = np.zeros(shape=(len(x_test),), dtype=np.float64)
  csim(x_train, a_train)
  csim(x_test, a_test)

  return a_train, a_test

tave = np.zeros(shape=(10,))
eave = np.zeros(shape=(10,))
for i, (train_index, test_index) in enumerate(kfold.split(v_train)):
  print(i)
  a_train, a_test = train(v_train[train_index], v_train[test_index])
  tave[i] = ss.pearsonr(a_train, y_train[train_index])[0]
  eave[i] = ss.pearsonr(a_test, y_train[test_index])[0]
print('kfold train ave')
print(np.mean(tave))
print('kfold test ave')
print(np.mean(eave))

a_train, a_test = train(v_train, v_test)
print(ss.pearsonr(a_train, y_train))
print(ss.pearsonr(a_test, y_test))

plt.scatter(y_train, a_train)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()
plt.scatter(y_test, a_test)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

import numpy.polynomial.polynomial as poly
# Fit polynomial model to output score
poly_coeff = poly.polyfit(a_train, y_train, 7)
s_train = poly.polyval(a_train, poly_coeff)
# Apply it to test data
s_test = poly.polyval(a_test, poly_coeff)
print(ss.pearsonr(s_train, y_train))
print(ss.pearsonr(s_test, y_test))

plt.scatter(y_train, s_train)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()
plt.scatter(y_test, s_test)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

x = np.arange(50)/10
plt.plot(x, poly.polyval(x, poly_coeff))

poly_coeff

x_train = np.zeros(shape=(len(x_train), 600))
for i, pair in enumerate(v_train):
  x_train[i] = np.concatenate((v_train[i][0], v_train[i][1]))

x_test = np.zeros(shape=(len(x_test), 600))
for i, pair in enumerate(v_test):
  x_test[i] = np.concatenate((v_test[i][0], v_test[i][1]))

x_train

y_train /= 5
y_train

seq = models.Sequential()
seq.add(layers.Dense(600, activation='sigmoid', input_shape=(600,)))
seq.add(layers.Dense(600, activation='sigmoid'))
seq.add(layers.Dense(1, activation='sigmoid'))
seq.compile(optimizer=optim.RMSprop(lr=0.001), loss=losses.mean_squared_logarithmic_error, metrics=['mae'])
seq.summary()

history = seq.fit(x_train, y_train, batch_size=25, epochs=100, validation_split=0.1, verbose=1)

hvmae = history.history['val_mean_absolute_error']
hvloss = history.history['val_loss']
hloss = history.history['loss']
hmae = history.history['mean_absolute_error']
plt.plot(np.arange(len(hloss)), hloss, 'g-')
plt.plot(np.arange(len(hvloss)), hvloss, 'r-')

p_test = seq.predict(x_test)
p_test = np.array([ t[0] for t in p_test ])
p_test *= 5
ss.pearsonr(p_test, y_test)

plt.scatter(y_test, p_test)
plt.xlabel('actual')
plt.ylabel('predicted')

p_train = seq.predict(x_train)
p_train = np.array([ t[0] for t in p_train ])
p_train *= 5
ss.pearsonr(p_train, y_train)

plt.scatter(y_train, p_train)
plt.xlabel('actual')
plt.ylabel('predicted')