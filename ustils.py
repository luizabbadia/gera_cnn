# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:22:40 2020

@author: luiza

## Este módulo realiza as seguintes funções:
1 - Realça frames escurecidos.
2 - Realiza um blur
3 - Cria foto desfocada
4 - Cria foto enhanced                
5 - Transforma imagem em Gray scale 
6 - Cria sets com 80%, 60% e 40% do original.
Ou seja o script realca multiplica por 9 o set de fotos originais           
7 - Caso a flag extras =1 o script transforma todas as fotos e não apenas o grupo 
original, ou seja, multiplica em 33 vezes o set original 
*******************************************************************************
8 - Nivela a quantidade de imagens das pastas pela quantidade de imagens da
pasta com o menor número, removendo as imagens necessárias em ordem aleatoria
para balancear o set     

"""
import numpy as np
import os
#import random
#import pandas as pd
#import shutil
import cv2
#import imutils
#import os
import sys

def reduz(img):
    imgshape = img.shape
    imgx = imgshape[0]
    imgy = imgshape[1]
    
    # Reduz em 80%
    dimx = int(imgx*.80)
    dimy = int(imgy*.80)
    img6 = img
    im6 = cv2.resize(img6, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)
                    
    # Reduz em 60%
    dimx = int(imgx*.60)
    dimy = int(imgy*.60)
    img7 = img
    im7 = cv2.resize(img7, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)

    # Reduz em 40%
    dimx = int(imgx*.40)
    dimy = int(imgy*.40)
    img8 = img
    im8 = cv2.resize(img8, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)

    return im6,im7,im8

def flipa(img):
	imfv = cv2.flip(img, 0)
	imfh = cv2.flip(img, 1)
	imfvh = cv2.flip(img, -1)
	return imfv,imfh,imfvh

def fliph(img):
	imrcw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	imrccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	return imrcw, imrccw

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def realca(path,extras,roti):

    # Coloca os diretorios em uma lista    
    lista = os.listdir(path)  # list of subdirectories and files
    
    # Extrai foto a foto das pastas
    for folder in lista:   
        print('*** Aplicando realce na pasta => ', folder)
        pasta =path + '/' + folder 
        for file in os.listdir(pasta):
            if file.endswith(".jpg"):
                #print(os.path.join(pasta, file))
                # Manipula as imagens    
                pass
                imagem = os.path.join(pasta, file)
                img = cv2.imread(imagem)                
               
                ## Realiza a manipulação das imagens
                # Para realçar os frames escuros
                hist,bins = np.histogram(img.flatten(),256,[0,256])
                
                cdf = hist.cumsum()
                #cdf_normalized = cdf * hist.max()/ cdf.max()
                
                cdf_m = np.ma.masked_equal(cdf,0)
                cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
                cdf = np.ma.filled(cdf_m,0).astype('uint8')
                
                
            	
				#gamma = 1.5                                  # change the value here to get different result				
				#im0 = adjust_gamma(original, gamma=gamma)


                kernel = np.ones((5,5),np.float32)/25
                # Cria foto desfocada
                
                im1 = cv2.filter2D(img,-1,kernel)
                
                # Cria foto enhanced
                im2 = cdf[img] 
                
                # Median filtering
                #im3 = cv2.medianBlur(img,5)
                gamma = 0.5  
                im3 = adjust_gamma(img, gamma=gamma)

                # Bilateral filtering
                gamma = 1.8  
                im4 = adjust_gamma(img, gamma=gamma)
                
                # Gray scale
                im5 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # Acha as dimensões               
               
                imgshape = img.shape
                imgx = imgshape[0]
                imgy = imgshape[1]
                
                # Reduz em 80%
                dimx = int(imgx*.80)
                dimy = int(imgy*.80)
                img6 = img
                im6 = cv2.resize(img6, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)
                                
                # Reduz em 60%
                dimx = int(imgx*.60)
                dimy = int(imgy*.60)
                img7 = img
                im7 = cv2.resize(img7, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)

                # Reduz em 40%
                dimx = int(imgx*.40)
                dimy = int(imgy*.40)
                img8 = img
                im8 = cv2.resize(img8, dsize=(dimy, dimx), interpolation=cv2.INTER_LINEAR)

                # Salva na pastas correspondentes 
                # rotate = 1
                # desfocada
                prefixo = '/d'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im1)
                if extras ==1:
                	#return imfv,imfh,imfvh
                    im9,im10,im11 = flipa(im1)
                    prefixo9 = prefixo +'9dfv'                                
                    foto = pasta + prefixo9 + file     
                    cv2.imwrite(foto,im9)
                    prefixo10 = prefixo +'10dfv'                                
                    foto = pasta + prefixo10 + file     
                    cv2.imwrite(foto,im10)
                    prefixo11= prefixo +'11dfv'                                
                    foto = pasta + prefixo11 + file     
                    cv2.imwrite(foto,im11)

               
                # enhanced    
                prefixo = '/e'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im2)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im12,im13,im14 = flipa(im2)
                    prefixo12 = prefixo +'12dfv'                                
                    foto = pasta + prefixo12 + file     
                    cv2.imwrite(foto,im12)
                    prefixo13 = prefixo +'13dfv'                                
                    foto = pasta + prefixo13 + file     
                    cv2.imwrite(foto,im13)
                    prefixo14= prefixo +'14dfv'                                
                    foto = pasta + prefixo14 + file     
                    cv2.imwrite(foto,im14)

                
                # medium filter
                prefixo = '/m'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im3)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im15,im16,im17 = flipa(im3)
                    prefixo15= prefixo +'15dfv'                                
                    foto = pasta + prefixo15 + file     
                    cv2.imwrite(foto,im15)
                    prefixo16 = prefixo +'16dfv'                                
                    foto = pasta + prefixo16 + file     
                    cv2.imwrite(foto,im16)
                    prefixo17= prefixo +'17dfv'                                
                    foto = pasta + prefixo17 + file     
                    cv2.imwrite(foto,im17)

                # bilateral filter
                prefixo = '/b'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im4)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im18,im19,im20 = flipa(im4)
                    prefixo18= prefixo +'18dfv'                                
                    foto = pasta + prefixo18 + file     
                    cv2.imwrite(foto,im18)
                    prefixo19 = prefixo +'19dfv'                                
                    foto = pasta + prefixo19 + file     
                    cv2.imwrite(foto,im19)
                    prefixo20= prefixo +'20dfv'                                
                    foto = pasta + prefixo20 + file     
                    cv2.imwrite(foto,im20)

                # gray
                prefixo = '/g'
                foto = pasta + prefixo + file               
                cv2.imwrite(foto,im5)
                if extras ==1:
                    
                    #return imfv,imfh,imfvh
                    im21,im22,im23 = flipa(im5)
                    prefixo21= prefixo +'21dfv'                                
                    foto = pasta + prefixo21 + file     
                    cv2.imwrite(foto,im21)
                    prefixo22 = prefixo +'22dfv'                                
                    foto = pasta + prefixo22 + file     
                    cv2.imwrite(foto,im22)
                    prefixo23= prefixo +'23dfv'                                
                    foto = pasta + prefixo23 + file     
                    cv2.imwrite(foto,im23)
                # reduced 80%
                prefixo = '/r8'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im6)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im24,im25,im26 = flipa(im6)
                    prefixo24= prefixo +'24dfv'                                
                    foto = pasta + prefixo24 + file     
                    cv2.imwrite(foto,im24)
                    prefixo25 = prefixo +'25dfv'                                
                    foto = pasta + prefixo25 + file     
                    cv2.imwrite(foto,im25)
                    prefixo26= prefixo +'26dfv'                                
                    foto = pasta + prefixo26 + file     
                    cv2.imwrite(foto,im26)

                # reduced 60%
                prefixo = '/r6'
                foto = pasta + prefixo + file                
                cv2.imwrite(foto,im7)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im27,im28,im29 = flipa(im7)
                    prefixo27= prefixo +'27dfv'                                
                    foto = pasta + prefixo27 + file     
                    cv2.imwrite(foto,im27)
                    prefixo28 = prefixo +'28dfv'                                
                    foto = pasta + prefixo28 + file     
                    cv2.imwrite(foto,im28)
                    prefixo29= prefixo +'29dfv'                                
                    foto = pasta + prefixo29 + file     
                    cv2.imwrite(foto,im29)
                # reduced 40%
                prefixo = '/r4'
                foto = pasta + prefixo + file
                cv2.imwrite(foto,im8)
                if extras ==1:
                    #return imfv,imfh,imfvh
                    im30,im31,im32 = flipa(im8)
                    prefixo30= prefixo +'30dfv'                                
                    foto = pasta + prefixo30 + file     
                    cv2.imwrite(foto,im30)
                    prefixo31 = prefixo +'31dfv'                                
                    foto = pasta + prefixo31 + file     
                    cv2.imwrite(foto,im31)
                    prefixo32= prefixo +'32dfv'                                
                    foto = pasta + prefixo32 + file     
                    cv2.imwrite(foto,im32)

    
    if roti == 1:
	    for folder in lista:   
	        print('*** Aplicando realce na pasta => ', folder)
	        pasta =path + '/' + folder 
	        for file in os.listdir(pasta):
	            if file.endswith(".jpg"):
	                # Manipula as imagens    
	                pass
	                imagem = os.path.join(pasta, file)
	                #print('imagem',imagem)
	                #return imrcw, imrccw
	                img = cv2.imread(imagem)
	                im33,im34 = fliph(img)
	                prefixo33= prefixo +'33dfv'
	                foto = pasta + prefixo33 + file
	                cv2.imwrite(foto,im33)
	                prefixo34= prefixo +'34dfv'
	                foto = pasta + prefixo34 + file
	                cv2.imwrite(foto,im34)
                    
def nivelar(path,extra): 
#################################################################################
## ESTE PROGRAMA FAZ O BALACEAMENTO DO DATASET NIVELANDO O NÚMERO DE ARQUIVOS   #
## DE TODAS AS PASTAS PELA QUANTIDADE MENOR DOS  ARQUIVOS                       #
## PARA TAL ELE PRIMEIRO RENOMEIA E RENUMERA OS ARQUIVOS PARA QUE POSSAM SER    #
## EXCLUÍDOS RANDOMICAMENTE                                                     # 
#################################################################################      
    mydir = path
    print(mydir)
    count = 0     
    folder_names = []     
    #################################################################
    ## Prepara para nomear os arquivos e numerá-los sequencialmente #
    ## para realizar futuramente o nivelamento                      #
    #################################################################
    # Extrai os diretorios
    #diretorio = "datawork"
    contador = 0
    num_files = 0
    for entry_name in os.listdir(mydir):
        entry_path = os.path.join(mydir, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)
            n_folders = len(folder_names)
            #n_pastas = n_folders

    if len(folder_names) ==0:
        print("Sem extracao...diretorio filmes vazio...")
        sys.exit()      
    
    for dir in  folder_names:    
        contador += 1        
        #print('Num de folders => ',folder_count)
        in_directory = mydir + "/" + dir    
        ### INICIA
        directory = os.fsencode(in_directory)
        # Faz iteração sobre todos as fotos do diretório assinalado
        dire = os.listdir(directory)    
        dire1 = len(dire)
        num_files = num_files + dire1   
        
        for file in os.listdir(directory):
            filename = os.fsdecode(file)        
            dire1 -= 1
        
        
        """
        for count, filename in enumerate(os.listdir(in_directory)): 
            for dir in  folder_names:    
                contador += 1        
                #print('Num de folders => ',folder_count)
                in_directory = mydir + "/" + dir    
                ### INICIA
                directory = os.fsencode(in_directory)
                # Faz iteração sobre todos as fotos do diretório assinalado
                dire = os.listdir(directory)    
                dire1 = len(dire)
                num_files = num_files + dire1  



        #print('Dir => ' ,dir)
        #print('in_directory',in_directory)     
  
        """
        for count, filename in enumerate(os.listdir(in_directory)):               
            src = in_directory + '/' + filename 
            dst = in_directory + "/" + str(count) + ".jpg"        
            try:
                os.rename(src, dst) 
            except FileExistsError: 
                pass                    
        


    ####################################################################
    ## Com os arquivos renomeados e renumerados passamos a operação de # 
    ## nivelamento para que as pastas contenham um número igual de     #
    ## arquivos, nivelados pelo menor número e assim as pastas ficam   #
    ## balanceadas para a criação da rede cnn                          #                       
    ####################################################################
    #mydir = path
    #print(mydir)
    count = 0 
    folder_names = []    
    contador = 0
    num_files = 0 
    remover = 1 
    flag = 0
    while remover != 0:
    
        folder_names = []
        objetos = []
        quantidades= []
        #exclui = []
        
        #Para achar diretorios
        for entry_name in os.listdir(mydir):
            entry_path = os.path.join(mydir, entry_name)
            if os.path.isdir(entry_path):
                folder_names.append(entry_name) 
        try:
            #print(folder_names[0])
            pass
        except IndexError:
            print("Diretorio Creator vazio !!!!")
            exit
        nfolders = len(folder_names)
        #print(nfolders )
        #print("------------------------------------")     
        for dir in  folder_names:
            #print(dir)    
            cpt= sum([len(files) for r, d, files in os.walk(mydir + "/" + dir)])    
            #print(cpt)
            objetos.append(dir)
            quantidades.append(cpt)        
     
        #Extrai o maior valor de fotos
        maximo = np.argmax(quantidades)
        maior = quantidades[maximo]
        if flag ==0:
            print("Maior ",maior)
          
        #Extrai o menor valor de fotos
        minimo = np.argmin(quantidades)
        menor = quantidades[minimo]       
        
        #Apresenta a análise
        if flag ==0:               
        
                print( "************ INICIA ***************")
                print("*===============================================*")
                print("O programa nivela encontrou os seguintes dados: ")
                print("Pasta principal =>        : " + mydir) 
                print("Quantidade de subpastas => : " + str(nfolders)) 
        
        for x in range(nfolders):
            if flag ==0:
                print( objetos[x] , " fotos => :" , quantidades[x]) 
        
        if flag ==0:
                
                print("Maior quantidade de fotos => : " + str(maior)) 
                print("Menor quantidade de fotos => : " + str(menor)) 
                
                print("Quantidade de fotos a nivelar em " + str(menor)) 
        
        remover = 0
        for x in range(nfolders):
            if flag ==0:
                
                print( objetos[x] , " fotos => :" , quantidades[x]- menor) 
            remover = remover + (quantidades[x]- menor) 
        
        if flag ==0:
            print( "**** EXECUTANDO O NIVELAMENTO ****")
        flag = 1
        # Cria lista de fotos em ordem randomica para deletar
        #idx = np.random.randint(0, (quantidades[0]- menor) , 1)
        
        nome = ""
        for x in range(nfolders):
        #print(x)
            idx =[]
            tirar = quantidades[x]- menor
            idx = np.random.randint(tirar , quantidades[x] , tirar)
            #print(idx)
            leno = len(idx)
            for n in range(leno): 
                #print(idx[n])
                #idy =idx[0]
                nome = mydir + '/' + objetos[x]+ '/'+ str(idx[n]) + ".jpg"
                #print(nome)
                try:
                    os.remove(nome)
                except FileNotFoundError:
                    pass
            
        for count, filename in enumerate(os.listdir(in_directory)): 
            for dir in  folder_names:    
                contador += 1        
                #print('Num de folders => ',folder_count)
                in_directory = mydir + "/" + dir    
                ### INICIA
                directory = os.fsencode(in_directory)
                # Faz iteração sobre todos as fotos do diretório assinalado
                dire = os.listdir(directory)    
                dire1 = len(dire)
                num_files = num_files + dire1   
        
        for file in os.listdir(directory):
            filename = os.fsdecode(file)        
            dire1 -= 1
       
        for count, filename in enumerate(os.listdir(in_directory)): 
            #print('filename',filename)
            pass
  
    print("*===============================================*")
    print("O programa nivela finalizou com os seguintes dados: ")
    print("Pasta principal =>        : " + mydir) 
    print("Quantidade de subpastas => : " + str(nfolders)) 
    flag = 0
    for x in range(nfolders):       
        if flag ==0: 
            print("Maior quantidade de fotos => : " + str(maior)) 
            print("Menor quantidade de fotos => : " + str(menor)) 
            flag = 1
    
    
    return()
     