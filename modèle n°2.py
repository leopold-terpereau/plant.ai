# -*- coding: utf-8 -*-



import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.models import Model







i1 = 0
i2 = 0


dir1 = "dataset/plant_healthy"
dir2 = "dataset/plant_unhealthy"

for path in os.listdir(dir1):
    if os.path.isfile(os.path.join(dir1, path)):
        i1+= 1
        
for path in os.listdir(dir2):
    if os.path.isfile(os.path.join(dir2, path)):
        i2+= 1
        
i = i1 + i2
        
X_train = np.zeros((i, 256, 256, 3))
Y_train = np.zeros((i, 1))

k = 0


for path in os.listdir(dir1):
    if os.path.isfile(os.path.join(dir1, path)):
        Y_train[k] = 0
        image = Image.open(dir1 + "/" + path)
        X_train[k] = image
        k+=1
        
for path in os.listdir(dir2):
    if os.path.isfile(os.path.join(dir2, path)):
        Y_train[k] = 1
        image = Image.open(dir2 + "/" + path)
        X_train[k] = image
        k+=1

X_train = X_train/255.
        
print(X_train)
print(Y_train)




index = 2
#plt.imshow(X_train[index]) 
#plt.show()



def identity_block(X, f, filters, training=True, initializer=random_uniform):
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X) 

    return X



def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
   
    F1, F2, F3 = filters
    
    X_shortcut = X


    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X



def ResNet50(input_shape = (256, 256, 3)):
    
    
    X_input = Input(input_shape)

    
    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    
    X = convolutional_block(X, f = 3, filters = [64*2, 64*2, 256*2], s = 2)
    X = identity_block(X, 3, [64*2, 64*2, 256*2])
    X = identity_block(X, 3, [64*2, 64*2, 256*2]) 
    X = identity_block(X, 3, [64*2, 64*2, 256*2])
    
    
     
   
    X = convolutional_block(X, f = 3, filters = [64*2*2, 64*2*2, 256*2*2], s = 2)
    X = identity_block(X, 3, [64*2*2, 64*2*2, 256*2*2])
    X = identity_block(X, 3, [64*2*2, 64*2*2, 256*2*2]) 
    X = identity_block(X, 3, [64*2*2, 64*2*2, 256*2*2]) 
    X = identity_block(X, 3, [64*2*2, 64*2*2, 256*2*2])
    X = identity_block(X, 3, [64*2*2, 64*2*2, 256*2*2])

    
    X = convolutional_block(X, f = 3, filters = [64*2*2*2, 64*2*2*2, 256*2*2*2], s = 2)
    X = identity_block(X, 3, [64*2*2*2, 64*2*2*2, 256*2*2*2])
    X = identity_block(X, 3, [64*2*2*2, 64*2*2*2, 256*2*2*2]) 

    
    X = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(X)
    
    
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_initializer = glorot_uniform())(X)
    
    
    
    model = Model(inputs = X_input, outputs = X)

    return model




model = ResNet50(input_shape = (256, 256, 3))
print(model.summary())





model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

model.save("plant_model2.h5")
