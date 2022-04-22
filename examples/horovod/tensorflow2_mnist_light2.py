# Minimal mnist example with Tensorflow 2.0 available at xx
# This version is coming from horovod_tensorflow2_mnist_light.py
# It is with minor changes, to define explicitly the loss and opt

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train has shape (60000, 28,28) and y_train (600000)
# x_test has shape (10000, 28,28) and y_test (100000)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

loss = tf.losses.SparseCategoricalCrossentropy()
opt = tf.optimizers.Adam()

model.compile(optimizer=opt,
              loss = loss,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

