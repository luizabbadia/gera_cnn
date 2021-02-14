# Convolutional Neural Network

#################################################################################
## ESTE PROGRAMA EXAMINA UMA REDE CNN BASEADA NOS DADOS RECEBIDOS E GERA   #
## VÁRIOS RELATÓRIOS INCLUSVE UM ARQUIVO RESULTADO FINAL DAS REDES CRIADAS      # 
## CRIADO NA PASTA RAIZ CHAMADO AVALIA.CSV QUE SERVE PARA COMPARAÇÕES FUTURAS   #
## USAGE python3 gera_cnn.py -d 'diretório' -e [250] -b[32] -o {AdaGrad]        #
##               -1a [0] -1b [0] -2a [0] -2b [0] -n [128] -v [a] -p [20] -s [0] #                                                   #
#################################################################################

import matplotlib.pyplot as plt
import numpy as np
import splitfolders ## python -m pip install split_folders
import os
import random
import pandas as pd
import shutil
import cv2
import imutils
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import History, EarlyStopping
import keras
from keras.callbacks import ModelCheckpoint
import argparse
import winsound
import ustils
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
import sys

#para produzir o beep
duration = 10  # seconds
freq = 2000  # Hz
winsound.Beep(freq, duration)

# Imprime a versão do Tensorflow e Keras
import tensorflow as tf
print('**************************************')

print('TensorFlow Version ',tf.__version__)
print('Keras Version ',keras.__version__)
tfver = (tf.__version__)
kver = (keras.__version__)
print('**************************************')



############ TESTA O MODELO  ####################
#################################################
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# identical to the previous one

# Part 3 - Making new predictions
import numpy as np
import cv2
from keras.preprocessing import image
import pandas as pd
import os, os.path, shutil

#
######################################################
################## TESTES ############################
######################################################
#
show = 1
# Lê o arquivo conf
with open('conf.txt') as f:
    lines = [line.rstrip() for line in f]

cascade = lines[0]
print(cascade)
classes = lines[1]
print(classes)
modelo = lines[2]
print(modelo)
film = lines[3]


# Carrega o arquivo de classes
classesd = pd.read_csv(classes)
# Carrega o classificador haarcascade
classificador = cascade

cascade = cv2.CascadeClassifier(classificador)
# Define o modelo cnn
file = modelo
nbclasses = 2
bs = 32

print(classesd.head(5))
#classes = classesd.values.tolist()
classes = classesd['Classes'].to_list()
print (classes)
nbclasses = len(classes)
testdir = 'datatest'

# Define o modelo a carregar
perc = []
passo = 1
for rodada in range(passo):
    
    if rodada == 1:
        modelo = file
    else:
        modelo = file
    passo -=1
    print(" ###########################################################################")
    print('  Modelo em uso :', modelo , ' passo=', passo)
    print(" ###########################################################################")
    
    # Carrega o modelo    
    classifier = load_model(modelo)
    # Reseta a lista
    predicted = []  
    # Faz a iteração sobre todas as fotos da pasta test
    count = 1
    for count, filename in enumerate(os.listdir(testdir)):           
    
        imagem = testdir + '/' + filename
       
        # Lê e testa uma imagem específicade test
        #imagem = diretorio + '/tests/test/7.jpg'
        #####################################################
        """
        roi_color = cv2.imread(imagem)
        cv2.imshow("roi_color",roi_color)
        roi_color =cv2.cvtColor(roi_color , cv2.COLOR_BGR2RGB)
        roi_color = roi_color.astype(np.float32)
        test_image = cv2.resize(roi_color, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        test_image = image.array_to_img(test_image)
        """
        #####################################################
        test_image = image.load_img(imagem, target_size = (64, 64))
        #test_image = preprocessing(test_image)
        test_image = image.img_to_array(test_image)        
        test_image = np.expand_dims(test_image, axis = 0)        
        result = classifier.predict(test_image)        
        preds = classifier.predict(test_image, batch_size=bs).argmax(axis=1)
        predi = preds[0]
        # Predictions contém o array com as adivinhações
        # se são 3 classes o vetor tem 3 valores
        # para selecionar => predictions[0][classe de 0 a 2]
        # selecionamos o valor a testar
        
        # Acha valores do array 
        predictions = classifier.predict(test_image, batch_size=32)
        
        probw = predictions[0][preds]
        probx = probw[0]* 100
        probx = round(probx,1)
        
        prediction =""
        #training_set.class_indices
        index = 9
        if nbclasses == index:        
            if result[0][predi] > 1:
                prediction = classes[predi]
                
        index -=1  
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                            
        index -=1    
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                            
        index -=1   
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                            
        index -=1  
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                
        index -=1   
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                
        index -=1  
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
                
        index -=1   
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
        
        index -=1  
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
        """
        index =0   
        if nbclasses == index:
            if result[0][predi] > 1:
                prediction = classes[predi] 
        """
        probx = '{0:.1f}'.format(probx)
        print ( 'O objeto é :',classes[predi])
        print('Image:',imagem ,' Res :',
              result,' Prob:',probx)
        #print(prediction)
        if show ==1:
            img =cv2.imread(imagem,1)
            cv2.putText(img, classes[predi], ( 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 200, 210), 2) 
            cv2.putText(img, classes[predi], ( 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 10, 210), 2) 
            cv2.imshow("guess",img)
            cv2.waitKey() 
        
        predic = classes[predi]
        predicted.append(predi)


        
  
    # Cria o prognóstico 
    print()
    print(" #####################################################################")
    print(' *********************  Prognóstico do Teste *************************')
    print(" #####################################################################")
    print()
    print('######################################################################')
    print('Modelo em uso :', modelo )
    print('######################################################################')
    
    final =[]
    quant = 0
    
    nprog = len(predicted_class_indices)
    for y in range(nprog):
        if predicted[y] == predicted_class_indices[y]:
            final.append(1)
            quant +=1
        else:
            final.append(0)
            
    print('Predict  :',predicted)
    print('Prognost :',final)
    
    #print('Confusion matrix \n',confusion_matrix(predicted, final))
    
   

    
    # Plot Confusion matrix
    graph_file =   folderfiles + '/' +  nome_modelo +'_cf_train_matrix.png'
    plt.figure(figsize=(10,8))
    #cf_train_matrix = confusion_matrix(predicted, final)
    
    #sns.heatmap(cf_train_matrix, annot=True, fmt='d')

    #fig = plt.figure()
    #fig.savefig(graph_file, dpi=fig.dpi)



    perci = (quant/nprog )*100 
    perci = '{0:.1f}'.format(perci)
    
    perc.append(perci)
    print('Quant :',quant,' Perc :', str(perci) + '%')   
    
# Extrai dados
eval_loss = eval[0]
eval_acc = eval[1]
best_mod = perc[0]
norm_mod = perc[1]
perc0 = perc[0]
perc1 = perc[1]      
    
# Salva o prognóstico dentro de arquivo csv
arquivo_prognostico = folderfiles + '/' +  nome_modelo + '_prognostico.csv'

prog = pd.DataFrame({'Quant': quant, 'Perc0':perc0, 'Perc1':perc0}, index=[0])
prog.to_csv(arquivo_prognostico,index=False)


# Verifica se existe um arquivo com o nome do diretorio, se existe, 
# lê o arquivo e faz o append
nome_avalia = diretorio +'/' + 'avalia.csv'
if (os.path.isfile(nome_avalia)):  
    aval = pd.read_csv(nome_avalia)
    
    aval = aval.append({'Modelo': nome_modelo, 
                        'Epochs_ran':number_of_epochs_it_ran,
                        'Loss':eval_loss,
                        'Acc': eval_acc, 
                        'Best_Model': best_mod,
                        'Normal_Model':norm_mod},
                        ignore_index=True)
    
    # Elimina duplicatas
    #aval = aval.drop_duplicates()
# Se não existe, cria o novo diretorio e grava os dados
else:        
    aval = pd.DataFrame({'Modelo': nome_modelo, 
                         'Epochs_ran':number_of_epochs_it_ran,
                         'Loss':eval_loss,
                         'Acc': eval_acc,
                         'Best_Model': str(best_mod),
                         'Normal_Model':str(norm_mod)}, 
                        index=[0])

arquivo_avalia = nome_avalia
print('*************************************************************************')
aval.to_csv(arquivo_avalia,index=False)
print(aval.head())
print('*************************************************************************')
"""
cv2.destroyAllWindows()
