# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import tensorflow as tf


#Déclaration des dossiers contenant les images
dir1 = "dataset/plant_type_healthy"
dir2 = "dataset/plant_type_unhealthy"


#Comptage du nombres d'images aptes pour l'entraînement du CNN
i1 = 0
i2 = 0
for path in os.listdir(dir1):
    if os.path.isfile(os.path.join(dir1, path)):
        i1+= 1
        
for path in os.listdir(dir2):
    if os.path.isfile(os.path.join(dir2, path)):
        i2+= 1
        
i = i1 + i2
        
#Initialisation des vecteurs d'entrées et de sorties
X_train = np.zeros((i, 256, 256, 3))
Y_train = np.zeros((i, 1))

#Constitution des matrices d'entrée et de sortie, de manière itérative
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

#Normalisation
X_train = X_train/255.
        
#print(X_train)
#print(Y_train)

#index = 2
#plt.imshow(X_train[index]) 
#plt.show()



#Définition de la fonction qui instancie un CNN
def plantModel():
   
    model = tf.keras.Sequential([
            
        
        tf.keras.layers.ZeroPadding2D(padding=(3,3), input_shape=(256, 256, 3)), #Première couche
        tf.keras.layers.Conv2D(32, (7,7), strides=(1,1)),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid") #Dernière couche
        
        
            
        ])
    
    return model



#Instanciation du modèle 
plant_model = plantModel()
    

#Compilation 
plant_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

#Résumé des caractéristiques de notre réseau de neurones
plant_model.summary()


#Phase d'entraînement
plant_model.fit(X_train, Y_train, epochs=10)

#Enregistrement du modèle dans le dossier courant
plant_model.save("plant_type.h5")
