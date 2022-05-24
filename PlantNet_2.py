
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
def plantModel(input_shape):
    

    input_img = tf.keras.Input(shape=input_shape)
    
    
    Z1=tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(1,1), padding='same')(input_img)
    A1=tf.keras.layers.ReLU()(Z1)
    P1=tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=(8,8), padding='same')(A1)
    Z2=tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='same')(P1)
    A2=tf.keras.layers.ReLU()(Z2)
    P2=tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=(4,4), padding='same')(A2)
    F=tf.keras.layers.Flatten()(P2)
    outputs=tf.keras.layers.Dense(units=2, activation='softmax')(F)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
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