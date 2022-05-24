# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import tensorflow as tf


#Déclaration des dossiers contenant les images de plantes en bonne et mauvaise santé
train_dir = "train"


Size = 256 #taille des images d'entrées



#Génération des tenseurs d'images pour l'entraînment
train_generator=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0).flow_from_directory(train_dir,batch_size=16,target_size=(Size,Size))




#Définition de la fonction qui instancie un CNN
def plantModel():
   
    model = tf.keras.Sequential([
            
        
        tf.keras.layers.ZeroPadding2D(padding=(3,3), input_shape=(256, 256, 3)), #Première couche
        tf.keras.layers.Conv2D(32, (7,7), strides=(1,1)),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation="softmax") #Dernière couche
        
        
            
        ])
    
    return model



#Instanciation du modèle 
plant_model = plantModel()
    

#Compilation 
plant_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#Résumé des caractéristiques de notre réseau de neurones
plant_model.summary()


#Phase d'entraînement
plant_model.fit(train_generator, epochs=10)

#Enregistrement du modèle dans le dossier courant
plant_model.save("plant_type.h5")