import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Vygenerovanie jednoduchých dát
X_train = np.array([1, 2, 3, 4], dtype=float)
y_train = np.array([2, 4, 6, 8], dtype=float)

# Definícia modelu
model = Sequential()
model.add(Dense(units=1, input_shape=[1]))

# Kompilácia modelu
model.compile(optimizer='sgd', loss='mean_squared_error')

# Tréning modelu
model.fit(X_train, y_train, epochs=500)

# Predikcia na nových dátach
print(model.predict(np.array([10.0, 20, 55, 1234, 344456754. -3476])))
