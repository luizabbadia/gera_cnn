# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:01:57 2020

@author: luiza
"""


#########################################################
#################### TESTA O MODELO  ####################
# Testa o modelo contra as imagens contidas na pasta    #
# imagens_teste                                         #
# definir classes e modelo nas linhas 30 e 32           #
#########################################################
#########################################################

#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#
#from keras.layers import Dense
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
import os
#############################################


pasta = 'vernier2'
# Carrega o arquivo de classes
classesd = pd.read_csv('classes_folder/'+ pasta + '_classes.csv')
classes = classesd['Classes'].to_list()
modelo= 'model_folder/'+ pasta + '_32_128_p20_a_best_model.h5'
testdir = pasta + '/tests/test'
bs = 32
nbclasses = int(len(classes)) 
#
######################################################
################## TESTES ############################
######################################################
#
# Define o modelo a carregar
perc = []
passo = 2

    

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
    print(prediction)
    prob = probx + "%"
    
    img =cv2.imread(imagem,1)
    cv2.putText(img, classes[predi], ( 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
    cv2.putText(img, classes[predi], ( 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
    cv2.putText(img, classes[predi], ( 5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
    #cv2.putText(img, prob, ( 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 200, 210), 2) 
    cv2.imshow("guess",img)
    cv2.waitKey() 
    cv2.destroyAllWindows()  
    
    predic = classes[predi]
    predicted.append(predi)
    print('FIM')
cv2.destroyAllWindows()