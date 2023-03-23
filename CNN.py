import os 

import tensorflow as tf
from keras import layers, regularizers
from keras.datasets import cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def my_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(32, 3, kernel_regularizer=regularizers.L2(0.01))(inputs)
    
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    x = layers.MaxPool2D()(x)
    
    x = layers.Conv2D(64, 5, padding='same', kernel_regularizer=regularizers.L2(0.01))(x)
    
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    x = layers.Conv2D(128, 3, kernel_regularizer=regularizers.L2(0.01))(x)
    
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.01))(x)
    
    x = layers.Dropout(0.5)
    outputs = layers.Dense(10)(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    return model
      
    

# model = tf.keras.Sequential(
#     [ 
#         tf.keras.Input(shape=(32, 32, 3)),
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, 3, activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, 3, activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10)
#     ]
# )

model = my_model()

print(model.summary())
model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = Adam(learning_rate=0.0003),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
print(model.summary())