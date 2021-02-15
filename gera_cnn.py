# Convolutional Neural Network

#################################################################################
## ESTE PROGRAMA EXECUTA CRIA UMA REDE CNN BASEADA NOS DADOS RECEBIDOS E GERA   #
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


######################################################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir_", required=True,
                help="diretorio fonte")
ap.add_argument("-e", "--epoch_",type=int, required=False,
    default = 400, help="n.º epochs")
ap.add_argument("-b", "--bs_",type=int, required=False,
    default = 32, help="batch size")
ap.add_argument("-o", "--opt_", required=False,
    default = 'AdaGrad', help="optimizer")
ap.add_argument("-1a", "--1a_",type=int, required=False,
    default = 0, help="uma camada tipo 1")
ap.add_argument("-2a", "--2a_",type=int, required=False,
    default = 0, help="duas camada tipo 1")
ap.add_argument("-1b", "--1b_",type=int, required=False,
    default = 0, help="uma camada tipo 2")
ap.add_argument("-2b", "--2b_",type=int, required=False,
    default = 0, help="duas camada tipo 2")
ap.add_argument("-n", "--densa_",type=int, required=False,
    default = 128, help="dense layer")
ap.add_argument("-v", "--ver_", required=False,
    default = 'a',help="versão")
ap.add_argument("-r", "--realce_",type=int,required=False,
    default = 1,help="bypass realça")
ap.add_argument("-p", "--pac_",type=int, required=False,
    default = 1, help="paciencia")
ap.add_argument("-s", "--show_",type=int, required=False,
    default = 0, help="show teste")
ap.add_argument("-x", "--extra_",type=int, required=False,
    default = 0, help="extra processing")
ap.add_argument("-t", "--test_",type=int, required=False,
    default = 1, help="imagens para teste")
ap.add_argument("-y", "--rot_",type=int, required=False,
    default = 0, help="rotate")

args = vars(ap.parse_args())
 
###############################################################################
###############################################################################
##################    DADOS DE ENTRADA  #######################################
###############################################################################
print('INFO: define o diretorio das classes e modelo= ' ,args['dir_'])
diretorio = args['dir_']
print('INFO: define quantidade de epochs= ',args['epoch_'])
epoch = args['epoch_']
print('INFO: define o batch size= ',args['bs_'])
bs = args['bs_']
print('INFO: define o Optmizer= ',args['opt_'])
opt = args['opt_']
print('INFO: define as camadas Conv2D tipo 1a= ',args['1a_'])
layer1a = args['1a_']
print('INFO: define as camadas Conv2D tipo 1b= ',args['1b_'])
layer1b = args['1b_']
print('INFO: define as camadas Conv2D tipo 2a= ',args['2a_'])
layer2a = args['2a_']
print('INFO: define as camadas Conv2D tipo 2b= ',args['2b_'])
layer2b = args['2b_']
print('INFO: define a camada dense= ',args['densa_'])
densa = args['densa_']
print('INFO: define versão= ',args['ver_'])
version = args['ver_']
print('INFO: define nome do modelo= ',args['dir_'])
nome_modelo = args['dir_']
print('INFO: define o nível de paciencia= ',args['pac_'])
paciencia = args['pac_']
print('INFO: define se realca as imagens= ',args['realce_'])
realc =  args['realce_']
print('INFO: define se realça com extras= ',args['extra_'])
extras =  args['extra_']
print('INFO: define se rotaciona imagens= ',args['rot_'])
rota =  args['rot_']
# Seleciona best ou normal
best = 1
print('INFO: define a mostra do teste= ',args['show_'])
show = args['show_']
print('INFO: define numero de imagens para teste= ',args['test_'])
t = args['test_']
#
##############################################################################
###############################################################################
###############################################################################
number_of_epochs_it_ran = 0
# Part 1 - Building the CNN

# Compõe o nome do modelo 
sufixo =''

if layer2a ==1:
    sufixo = sufixo + '2a_'
elif layer1a ==1:
    sufixo = sufixo + '1a_'
if layer2b ==1:
    sufixo = sufixo +'2b_'
elif layer1b ==1:
    sufixo = sufixo + '1b_' 
sufixo = sufixo + str(densa) + '_'
sufixo = sufixo + 'p'+ str(paciencia)+ '_'
sufixo = sufixo +  version

# Define o nome do modelo a salvar
nome_modelo = diretorio + '_' + str(bs) + '_' + sufixo

# Compõe o nome do modelo
modelo = diretorio + '/' + nome_modelo + '.h5' 

print('INFO: define nome do modelo= ',nome_modelo)

folderfiles = diretorio + '_files/' + nome_modelo
os.makedirs(folderfiles) 
# Este parâmetro define o arquivo onde os folders com imagens divididas em classes
# fica localizado e aonde ocorrerá o nivelamento e de onde sairão os arquivos 
# para a criação do modelo que serão copiados para o diretório output
mydir = "datawork"
print('INFO: define o diretorio source= ',mydir)


# Diretorio de teste
testdir = diretorio + "/tests/test"
   
###############################################
# Checa se já fez a fase "full" 
arquivo_full = diretorio + '/full.csv'
if (os.path.isfile(arquivo_full)):  
    full = 1
    # Se não existe, sinaliza para criar o arquivo full como flag
    # Extrai as classes , numero de classes e prognóstico
    classes =[]   
    # Se existe usa-o
    nome_classes = diretorio +'/' + 'classes.csv'      
    # Carrega o arquivo de classes
    classesd = pd.read_csv(nome_classes)
    print(classesd.head(5))    
    classes = classesd['Classes'].to_list()
    
else:    
    full = 0
#######################################
#######################################

# Ativa ou não meus plugins
if full ==0:    
    print('Criando o modelo pela primeira vez')
    # sinaliza com bip
    winsound.Beep(freq, duration)   
    # Cria o folder de files    
    out_directory = diretorio + '_files'
    #os.mkdir(out_directory)
    
    # Meus plugins
    ###############################################################################
    # NIVELAMENTO,SPLIT,CRIA TESTS FOLDERS,MOVE IMAGENS PARA TEST FOLDER,RENUMERA
    ###############################################################################
    #    
    # Extrai as classes , numero de classes e prognóstico    
    
    # Executa o nivelamento
    print( 'Executa o nivelamento dos folders')
    ustils.nivelar(mydir,extras)
    
    # Executa o enhancement das fotos criando fotos mais nétidas e mais desfocadas
    if realc ==1:
        print('Executa a manipulação das imagens')
        ustils.realca(mydir,extras,rota)
    
    # Executa o split dos folders
    print('Executa o split dos folders')
    splitfolders.ratio('datawork', output = diretorio, seed=1337, ratio=(.8, .2))
    
   
    classes =[]   
    # Se existe usa-o
    nome_classes = diretorio +'/' + 'classes.csv'
    if (os.path.isfile(nome_classes)):      
        # Carrega o arquivo de classes
        classesd = pd.read_csv(nome_classes)
        #print(classesd.head(5))    
        classes = classesd['Classes'].to_list()
        print('*** Pasta já existe, usando as classes existentes')   
       
    # Se não existe, cria o novo diretorio e grava os dados
    else: 
        # sinaliza com bip
        winsound.Beep(freq, duration)   
        print('*** Pasta nao existe, criando novo diretorio com as classes')    
        for entry_name in os.listdir(mydir):
            entry_path = os.path.join(mydir, entry_name)
            if os.path.isdir(entry_path):
                classes.append(entry_name)
                 
                arquivo_classes = nome_classes
                classi = pd.DataFrame(classes, columns=["Classes"])
                classi.to_csv(arquivo_classes, index=False)            
             
        arquivo_classes = diretorio +'/' + 'classes.csv'
        df = pd.DataFrame(classes, columns=["Classes"])
        df.to_csv(arquivo_classes, index=False)
    
        # salva as classes dentro do classes_folder para eventual uso nos sistemas especialistas
        arquivo_classes = 'classes_folder/' + diretorio +  '_' + 'classes.csv'
        df.to_csv(arquivo_classes, index=False)
        
    print (classes)        
    nbclasses = int(len(classes)) 
    maximo = nbclasses - 1 
    print(nbclasses)  
   
    # Verifica se existe um folder de test
    testsdir = diretorio + "/tests"
    testdir = diretorio + "/tests/test"
    if (os.path.isdir(testsdir)):  
        pass
    # Se não existe, cria o novo diretorio
    else:    
        os.mkdir(testsdir)  
        os.mkdir(testdir) 
    
    # Extrai arquivos aleatoriamente das pastas train e val para test
    print('*** Executa a criação do folder test e transfere fotos')
    prognostico =[]
    path = mydir 
    files = os.listdir(path)
    npick = t 
    print('NPICK=',npick)
    pick = 1    
    for folders in files:      
        path = mydir + '/' + folders
        #print(folders)    
        for x in os.listdir(path):
            random_filename = random.choice(os.listdir(path))
            #print(random_filename)
      
            try:
                src = mydir +'/'+ folders +'/'+ random_filename
                dst = diretorio + '/tests/test'
                shutil.move(src,dst) 
                #img1 = cv2.imread(src)             
                pick +=1  
                print(path,random_filename)
                if pick > npick:
                    pick=1
                    break
            # Para checar se já file existe no destino
            except OSError:
                pass
   
    # Renumera os arquivos do folder test
    print('*** Executa a renumeração da fotos no folder test')
    count = 1
    for count, filename in enumerate(os.listdir(testdir)): 
                #print('filename',filename)                 
                src = testdir + '/' + filename 
                dst = testdir + '/'  + str(count) + ".jpg"                 
                #print('src',src)
                #print('dst',dst)
                # rename() function will 
                # rename all the files 
                os.rename(src, dst) 
                
    # Salva o status de 'full' dentro de arquivo csv como 'flag' 
    # para sinalizar que já criou os arquivos
    print(' ***Executa a criação do arquivo full.csv para sinalização de etapa terminada')
    arquivo_full = diretorio + "/full.csv"
    full = pd.DataFrame({'Full': 'full'}, index=[0])
    full.to_csv(arquivo_full,index=False)
    
# Extrai as classes , numero de classes e prognóstico
classes =[]   
# Se existe usa-o
nome_classes = diretorio +'/' + 'classes.csv'
if (os.path.isfile(nome_classes)):      
    # Carrega o arquivo de classes
    classesd = pd.read_csv(nome_classes)
    #print(classesd.head(5))    
    classes = classesd['Classes'].to_list()
    print('*** Pasta já existe, usando as classes existentes')   
   
# Se não existe, cria o novo diretorio e grava os dados
else: 
    # sinaliza com bip
    winsound.Beep(freq, duration)   
    print('*** Pasta nao existe, criando novo diretorio com as classes')    
    for entry_name in os.listdir(mydir):
        entry_path = os.path.join(mydir, entry_name)
        if os.path.isdir(entry_path):
            classes.append(entry_name)
             
            arquivo_classes = nome_classes
            classi = pd.DataFrame(classes, columns=["Classes"])
            classi.to_csv(arquivo_classes, index=False)            
         
    arquivo_classes = diretorio +'/' + 'classes.csv'
    df = pd.DataFrame(classes, columns=["Classes"])
    df.to_csv(arquivo_classes, index=False)

    # salva as classes dentro do classes_folder para eventual uso nos sistemas especialistas
    arquivo_classes = 'classes_folder/' + diretorio +  '_' + 'classes.csv'
    df.to_csv(arquivo_classes, index=False)
    
print (classes)        
nbclasses = int(len(classes)) 
maximo = nbclasses - 1 
print(nbclasses)  

# Salva as versões de Tensorflow e Keras dentro do arquivo versions

arquivo_versions = diretorio + '/versions.csv'
tfver = (tf.__version__)
kver = (keras.__version__)
ver = pd.DataFrame({'Versao TensorFlow': tfver,' Versao Keras':kver}, index=[0])

ver.to_csv(arquivo_versions,index=False)


###############################################################################   
# Importing the Keras libraries and packages
# Responsável por 
from keras.models import Sequential
# Inicializa a neural network
from keras.layers import Conv2D
from keras.layers.core import Dropout
#from keras.utils import multi_gpu_model
#Esse módulo realiza a Convolução sendo 2D pois são imagens
#
from keras.layers import MaxPooling2D
#Esse é a fase Pooling
from keras.layers import Flatten
#Fase Flattening
from keras.layers import Dense
#Para adicionar a conexão 

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), 
                      activation = 'relu',
                      data_format='channels_first'))

if layer1a ==1:
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
if layer2a ==1:
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Pooling# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (3, 3)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
if layer1b ==1:
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
if layer2b ==1:
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    
# Pooling# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (3, 3)))

#Dropout
classifier.add(Dropout(0.20))
#####################
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

classifier.add(Dense( densa, activation = 'relu'))
#classifier.add(Dense(units = 128, activation = 'relu'))
#####################################
# units é o numero de saídas
classifier.add(Dense( nbclasses, activation = 'softmax')) 

classifier.compile(optimizer = opt, 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory= diretorio + '/train',
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=bs,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
val_generator = train_datagen.flow_from_directory(
    directory= diretorio + '/val',
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=bs,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory= diretorio + '/tests',
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
#
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
print('train_generator.n',train_generator.n)
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
print('val_generator.n ',val_generator.n, 'val_generator.batch_size ',val_generator.batch_size )
if val_generator.n < val_generator.batch_size:
    print('*** ERRO ****************  Val_gen.n error - Aumente a quantidade de imagens')
    sys.exit()

# Conforme https://keras.io/callbacks/
#es = EarlyStopping(monitor='val_loss', min_delta = 0,
                                        #patience=paciencia, verbose=1, mode='min', 
                                        #baseline=0.0)

es = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                    patience=paciencia,
                                    verbose=0, mode='auto')


mc = keras.callbacks.ModelCheckpoint(folderfiles + '/' +  nome_modelo + '_best_model.h5'
                     , monitor='val_loss', 
                     mode='min', save_best_only=True, verbose=0)

#ese = early_stopping_monitor.stopped_epoch

# Adicionado o callback early stop no fit
H = classifier.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,                    
                    epochs=epoch, verbose=1,
                    callbacks=[es,mc]                
)
number_of_epochs_it_ran = len(H.history['loss'])
#####################################################################################

plot_model(classifier, to_file = folderfiles + '/' +  nome_modelo + '_model_arch.png')
# Como em https://keras.io/visualization/

# Plot training & validation accuracy values
graph_file = folderfiles + '/' +  nome_modelo + '_acc_values_history.png'
fig = plt.figure()

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
"""
plt.show()
"""
fig.savefig(graph_file, dpi=fig.dpi)

# Plot training & validation loss values
graph_file =   folderfiles + '/' +  nome_modelo +'_loss_values_history.png'
fig = plt.figure()

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
#print(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
"""
plt.show()
"""
fig.savefig(graph_file, dpi=fig.dpi)
################################################################################
#########################################
## Seção de avaliação do modelo         #
#########################################
#
# Evaluate the model
eval = classifier.evaluate_generator(generator=val_generator,
                              steps=STEP_SIZE_VALID)

print("Eval :", eval)

# Test the model
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=classifier.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)
###############################################################################

# Salva parametros dentro do arquivo parametros.csv
arquivo_parametros = folderfiles + '/' +  nome_modelo + '_parametros.csv'
par = pd.DataFrame({'Diretorio': diretorio, 'Mydir':mydir,'NBclasses': nbclasses, 
                    'Optimizer': opt,'Epochs':epoch,'Batch_size':bs,'Modelo':modelo}, index=[0])
par.to_csv(arquivo_parametros,index=False)

# Salva as classes dentro do best/best_classes para uso nos sistemas especialistas
org =diretorio +'/' + 'classes.csv'
dest =  'best/best_class/' +  'class.csv'
shutil.copy(org , dest)

# Copia o best_model para o best/best_model para uso nos sistemas especialistas
org = folderfiles + '/' +  nome_modelo  + '_best_model.h5'
dest =  'best/best_model/' + 'best_model.h5'
shutil.copy(org , dest)

# Salva results dentro do arquivo results.csv
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
arquivo_result = folderfiles + '/' +  nome_modelo + '_results.csv'
results.to_csv(arquivo_result,index=False)

# Salva o eval dentro de arquivo csv
arquivo_eval = folderfiles + '/' +  nome_modelo + '_eval.csv'
ev = pd.DataFrame({'Eval': 'Eval', 'Value':eval})
ev.to_csv(arquivo_eval,index=False)

# Salva o early epoch stopped dentro de arquivo csv
arquivo_early = folderfiles + '/' +  nome_modelo + '_early.csv'
er = pd.DataFrame({'Early' : number_of_epochs_it_ran}, index=[0])
er.to_csv(arquivo_early,index=False) 

from keras.models import load_model
# salva o modelo dentro da pasta
#classifier.save(modelo)  # creates a HDF5 file 'my_model.h5'
modelo1 = folderfiles + '/' +  nome_modelo  + '.h5' 
classifier.save(modelo1)


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred1)

print(classifier.summary())
original_stdout = sys.stdout # Save a reference to the original standard output

with open(folderfiles + '/' +  nome_modelo + '_sumary.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(classifier.summary(), file=f)
    sys.stdout = original_stdout # Reset the standard output to its original value

#################################################
############ TESTA O MODELO  ####################
#################################################

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model


# identical to the previous one

# Part 3 - Making new predictions
import numpy as np
import cv2
from keras.preprocessing import image

#
######################################################
################## TESTES ############################
######################################################
#
# Define o modelo a carregar
perc = []
passo = 2
for rodada in range(passo):
    
    if rodada == 1:
        modelo = modelo1     
    else:
        modelo = folderfiles + '/' +  nome_modelo +'_best_model.h5'
    passo -=1
    print(" ###########################################################################")
    print('  Modelo em uso :', modelo , ' passo=', passo)
    print(" ###########################################################################")
    
    # Carrega o modelo    
    classifier = load_model((modelo), compile=False)
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
cv2.destroyAllWindows()
