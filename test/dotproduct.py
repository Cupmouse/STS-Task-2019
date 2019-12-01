import keras.models as models
import keras.layers as layers
import keras.optimizers as optims

import numpy as np
import scipy
import matplotlib.pyplot as plt

def itr_ndarray1(ndarr):
  for i in range(ndarr.shape[0]):
    yield ndarr[i]
def generate_y(x):
  y = np.zeros(shape=(x.shape[0],))
  for i, pair in enumerate(itr_ndarray1(x)):
    y[i] = scipy.spatial.distance.cosine(*pair)
  return y

x_train = np.concatenate((
    np.minimum(np.random.randint(size=(1500, 2, 50), low=0, high=6), 1),
    np.maximum(np.random.randint(size=(1500, 2, 50), low=0, high=6), 4)-4,
    np.random.random_sample(size=(1500, 2, 50))
    ), axis=0)
y_train = generate_y(x_train)
x_train = np.reshape(x_train, (-1, x_train.shape[1]*x_train.shape[2]))

x_test = np.concatenate((
    np.minimum(np.random.randint(size=(500, 2, 50), low=0, high=6), 1),
    np.maximum(np.random.randint(size=(500, 2, 50), low=0, high=6), 4)-4,
    np.random.random_sample(size=(500, 2, 50))
    ), axis=0)
y_test = generate_y(x_test)
x_test = np.reshape(x_test, (-1, x_test.shape[1]*x_test.shape[2]))

peep = (np.random.random_sample(size=(500, 2, 50))).reshape(-1)
plt.scatter(np.arange(150), peep[:150])

model = models.Sequential()
# model.add(layers.Dropout(rate=0.3, input_shape=(x_train.shape[1],)))
model.add(layers.Dense(50, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optims.RMSprop(lr=0.0001), loss='mean_squared_logarithmic_error', metrics=['mean_squared_error'])
model.summary()

h = model.fit(x_train, y_train, batch_size=250, epochs=50, validation_split=0.1, verbose=1)

hvalmse = h.history['val_mean_squared_error']
hmse = h.history['mean_squared_error']
plt.plot(np.arange(len(hvalmse)), hvalmse, 'b-')
plt.plot(np.arange(len(hmse)), hmse, 'y-')

p_test = model.predict(x_test)
print(scipy.stats.pearsonr(p_test.reshape(-1,), y_test))
plt.scatter(y_test, p_test)
plt.show()

p_train = model.predict(x_train)
print(scipy.stats.pearsonr(p_train.reshape(-1,), y_train))
plt.clf()
plt.scatter(y_train, p_train)
plt.show()

model.get_layer(index=0).get_weights()

model.get_layer(index=1).get_weights()

x = np.zeros(shape=(2, 50))
x[0][0] = 0
x[1][0] = 0
model.predict(x.reshape(1, -1))