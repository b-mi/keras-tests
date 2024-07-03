import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def reset_tf():
    tf.keras.backend.clear_session()
    for device in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, False)

def train_model(use_gpu):
    if use_gpu:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    reset_tf()  # Resetovanie grafu a pamäte pred každým trénovaním

    start_time = time.perf_counter()

    # Vygenerovanie väčšieho datasetu
    X_train = np.random.randn(100000).astype(float)
    y_train = 2 * X_train + 3
    X_val = np.random.randn(20000).astype(float)
    y_val = 2 * X_val + 3

    # Definícia zložitejšieho modelu
    model = Sequential()
    model.add(Dense(units=1280, activation='relu', input_shape=[1]))
    model.add(Dense(units=1280, activation='relu'))
    model.add(Dense(units=1280, activation='relu'))
    model.add(Dense(units=640, activation='relu'))
    model.add(Dense(units=640, activation='relu'))
    model.add(Dense(units=640, activation='relu'))
    model.add(Dense(units=1))

    # Kompilácia modelu
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Meranie času každej epochy
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_time_start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time_end = time.perf_counter()
            print(f"Trvanie epochy {epoch + 1}: {epoch_time_end - self.epoch_time_start:.2f} sekúnd")

    time_callback = TimeHistory()

    # Tréning modelu s validačnými dátami a menšou veľkosťou dávky
    try:
        history = model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val), batch_size=256, callbacks=[time_callback])
    except Exception as e:
        print(f"Chyba počas tréningu: {e}")

    # Predikcia na nových dátach
    prediction = model.predict(np.array([10.0, 123, -45]))
    print(prediction)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Trvanie funkcie: {duration} sekúnd")

    # Overenie podpory GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("GPU podpora je vypnutá, používa sa iba CPU.")
    else:
        print("GPU zariadenia:", gpus)

    # Vizualizácia stratovej funkcie
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Nastavte na '0' pre maximálne podrobnosti

# Tréning na GPU
# print("Tréning na GPU:")
# train_model(use_gpu=True)

# Tréning na CPU
print("Tréning na CPU:")
train_model(use_gpu=True)


