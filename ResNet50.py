import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,Dense,Flatten,ZeroPadding2D,BatchNormalization,Activation,Add,Input,Dropout,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50



#Crétaion des tenseurs d'images pour l'entraînement et la validation

train_datagenerator= ImageDataGenerator()

val_datagenerator=ImageDataGenerator()

path_train='/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

path_valid='/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

train= train_datagenerator.flow_from_directory(directory=path_train,batch_size=32,target_size=(256,256))

valid=val_datagenerator.flow_from_directory(directory=path_valid,batch_size=32,target_size=(256,256))







#importation du modèle non entraîné sur quoique ce soit
base_model_tf=ResNet50(include_top=False,weights=None,input_shape=(256,256,3),classes=38)


#ajout de fully-connected layers
def resnet50() :
    x = tensorflow.keras.Input(shape=(256, 256, 3))
    model_resnet=base_model_tf(x,training=False)
    model_resnet=GlobalAveragePooling2D()(model_resnet)
    model_resnet=Dense(128,activation='relu')(model_resnet)
    model_resnet=Dense(64,activation='relu')(model_resnet)
    model_resnet=Dense(38,activation='softmax')(model_resnet)


    model=Model(inputs=x,outputs=model_resnet)
    
    return model



#Instanciation du modèle 
model_main = resnet50()

#Compilation 
model_main.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Résumé des caractéristiques de notre réseau de neurones
model_main.summary()

#Phase d'entraînement
model_main.fit(train,validation_data=valid,epochs=11)

#Enregistrement du modèle dans le dossier courant
model_main.save('resnet50')