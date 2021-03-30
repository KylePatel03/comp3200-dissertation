import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

client_model = Sequential([
    Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten()
])
client_model.trainable = False

edge_model = Sequential([
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
])

main_model = Sequential([
    client_model,
    edge_model
])

main_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

