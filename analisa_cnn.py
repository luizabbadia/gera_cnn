# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:47:14 2020

@author: luiza
"""

#-*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:21:24 2019

@author: luiza
####################################################################################
## ESTE PROGRAMA EXECUTA PROGRAMAS EM SUA LISTA SEQUENCIALMENTE GERA RELATORIO     # 
## COM UM LOG DA EXECUCAO DOS PROGRAMAS CASO ELES GRAVEM SEUS SUMARIOS             #
## USAGE python analisa.cnn.py -d (diretorio) [-r [1]] [-p [20]] [-g [0]] [-s [0]] #
## EXEMPLO python analisa_cnn.py -d estist -r 0 -p 10 -g 1  -s 1                   #                                                                        #
####################################################################################

"""
import os
from datetime import datetime
import argparse
import shutil 
import pandas as pd
from threading import Timer

start_time = datetime.now() 
print('Iniciando', start_time  )

def estatistica(models):
    # Lê como dataframe
    #modelos = pd.read_csv(diretorio + '/avalia.csv')
    modelos = models
    print('#############################################################################################################################')
    print('#############################################################################################################################')
    ## para o sort uma coluna ascending e outra descending
    modelos.sort_values(['Best_Model', 'Loss'], ascending=[False,True ], inplace=True)
        
    # Salva a tabela
    modelos.to_csv(diretorio + '/avalia.csv',index=False)

    # Relê para reorganização
    modelos = pd.read_csv(diretorio + '/avalia.csv')
    print(modelos.head())
    
    # Imprime para visualização
    melhor_modelo = modelos.at[0,'Modelo'] + '.h5'
    
    return melhor_modelo

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir_", required=True,
                help="diretorio fonte")
ap.add_argument("-r", "--realce_",required=False,
    default = '1' ,help="bypass realça")
ap.add_argument("-p", "--paciencia_", required=False,
    default = '20', help="paciencia")
ap.add_argument("-s", "--solo_",type =int,required=False,
    default = '0' ,help="solo")
ap.add_argument("-f", "--full_",type =int,required=False,
    default = '0' ,help="full")
ap.add_argument("-x", "--extra_", required=False,
    default = '0', help="extra processing")
ap.add_argument("-n", "--new_", type =int, required=False,
    default = '0', help="novo set")
ap.add_argument("-t", "--test_",  required=False,
    default = '3', help="test set nb")
ap.add_argument("-o", "--opt_",  required=False,
    default = 'AdaGrad', help="optimizer")
ap.add_argument("-y", "--rot_",required=False,
    default = '0' ,help="rotate")

args = vars(ap.parse_args())

# Define o diretorio de saída
diretorio = args['dir_'] 
r = args['realce_']
p =  args['paciencia_']
x = args['extra_']
new = args['new_']
t = args['test_']
o = args['opt_']
y = args['rot_']

# Define se faz apenas a primeira passagem para fins de teste
solo = args['solo_']
if solo ==1:
    print ( '******   MODO SOLO ATIVADO !!!!   ******')

# Define se faz apenas a todas as configurações de cnn
full = args['full_']
if full ==1:
    print ( '******   MODO FULL ATIVADO !!!!   ******')

# Define se apaga o set original
if new ==1:
    print ( '******   MODO RESET ATIVADO !!!!   ******')

###############################################################################
# Apaga o diretorio datawork e copia tudo do diretório datatemp
# Isso é necessário pois assim sempre comeoçamos com um datawork com as imagens
# originais sem estarem modificadas pelo sistema pois as alterações tais como
# renumerações e realces são efetuadas no DATAWORK enquanto DATATEMP permanece
# inalterado.
###############################################################################
print('*** Deletando diretorio DATAWORK')
shutil.rmtree('datawork') 
print('*** Recria diretorio DATAWORK') 
print('*** Copiando diretorio DATATEMP para diretorio DATAWORK')
shutil.copytree('datatemp', 'datawork')

# Verifica se existe um arquivo com o nome do diretorio, se existe,apaga-o 
nome_avalia = diretorio +'/' + 'avalia.csv'
if (os.path.isfile(nome_avalia)):  
    aval = pd.read_csv(nome_avalia)
    print('*** Arquivo avalia.csv existente, deletando-o...')
    os.remove(nome_avalia)  
    
# Verifica se existe um arquivo com o nome do diretorio do set, se existe,apaga-o 
if new ==1:
    folder_sec =  diretorio 
    if (os.path.isdir(folder_sec)):  
        folder_sec
        print('*** Diretorio do set original existente, deletando-o...')
        shutil.rmtree(folder_sec, ignore_errors=True)  

# Verifica se existe um arquivo com o nome do diretorio secundário, se existe,apaga-o 
folder_sec =  diretorio + '_files'
if (os.path.isdir(folder_sec)):  
    folder_sec
    print('*** Diretorio secundario existente, deletando-o...')
    shutil.rmtree(folder_sec, ignore_errors=True)     

# Apaga e cria os subdiretórios 
folder_sec =  'best/best_class'
print('*** Apaga o sub-diretorio best_class')
shutil.rmtree(folder_sec)
print('*** Recria o sub-diretorio best_class')
os.mkdir(folder_sec) 
folder_sec =  'best/best_model'
print('*** Apaga o sub-diretorio best_model')
shutil.rmtree(folder_sec)
print('*** Recria o sub-diretorio best_model')
os.mkdir(folder_sec) 

# Compõe o batch
a0b0_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 0 -2b 0 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o + ' -y ' + y
a0b0_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 0 -2b 0 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b0_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 0 -2b 0 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b0_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 0 -2b 0 -n 256 '+ '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b0_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 0 -2b 0 '+ '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b0_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 0 -2b 0 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a0b1_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 1 -2b 0 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a0b1_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 1 -2b 0 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b1_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 1 -2b 0 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b1_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 1 -2b 0 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b1_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 1 -2b 0 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b1_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 1 -2b 0 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a0b2_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 0 -2b 1 '+ '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a0b2_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 0 -1b 0 -2b 1 -n 256 '  + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b2_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 0 -2b 1 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a1b2_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 1 -2a 0 -1b 0 -2b 1 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b2_128 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 0 -2b 1 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y
a2b2_256 = 'python gera_cnn.py -d ' + diretorio + ' -1a 0 -2a 1 -1b 0 -2b 1 -n 256 ' + '-r ' + r + ' -p ' + p + ' -x ' + x + ' -t ' + t + ' -o ' + o+ ' -y ' + y

# Executa o batch
os.system(a0b0_128)
if solo != 1:
    if full ==1:
        os.system(a0b0_256)
    os.system(a1b0_128)
    if full ==1:
        os.system(a1b0_256)
    os.system(a2b0_128)
    if full ==1:
        os.system(a2b0_256)
    os.system(a0b1_128)
    if full ==1:
        os.system(a0b1_256)
    os.system(a1b1_128)
    if full ==1:
        os.system(a1b1_256)
    os.system(a2b1_128)
    if full ==1:
        os.system(a2b1_256)
    os.system(a0b2_128)
    if full ==1:
        os.system(a0b2_256)
    os.system(a1b2_128)
    if full ==1:
        os.system(a1b2_256)
    os.system(a2b2_128)
    if full ==1:
     os.system(a2b2_256)

modelos = pd.read_csv(diretorio + '/avalia.csv')

#Aguarda para gravar infos
delay_in_sec = 1

#print(f'function called after {delay_in_sec} seconds')
#t = Timer(delay_in_sec, estatistica, [modelos])  # Hello function will be called 2 seconds later with [delay_in_sec] as the *args parameter
#t.start()  # Returns None
#print("Started")

melhor_modelo = estatistica(modelos)

# Salva a tabela
#modelos.to_csv(diretorio + '/avalia.csv',index=False)

# Imprime para visualização
#melhor_modelo = modelos.at[0,'Modelo'] + '.h5'

# Imprime para visualização
print (' O melhor model => ', melhor_modelo)
print('############################################################################################################################')
print('#############################################################################################################################')

# Salva o eval dentro de arquivo csv
arquivo_melhor = diretorio + '/melhor.csv'
ev = pd.DataFrame({'Melhor': melhor_modelo}, index=[0])
ev.to_csv(arquivo_melhor,index=False)

# copia o best_model para o model_folder
nome_modelo = melhor_modelo[:-3]
folderfiles = diretorio + '_files/' + nome_modelo
org = folderfiles + '/' +  nome_modelo  + '_best_model.h5'
dest =  'model_folder/'   #+ nome_modelo  + '_best_model.h5'
shutil.copy(org , dest)

time_elapsed = datetime.now() - start_time
print('Time_elapsed ',time_elapsed  )

# Salva o time_elapsed dentro de folder diretorio
arquivo_elapsed = diretorio + '/time_elapsed.csv'
td = pd.DataFrame({'Tempo decorrido': time_elapsed}, index=[0])
td.to_csv(arquivo_elapsed,index=False)

