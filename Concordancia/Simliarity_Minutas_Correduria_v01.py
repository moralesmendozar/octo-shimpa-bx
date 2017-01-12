# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:59:52 2017
# * Banco de México
# * Gerencia de Investigación del Sector Real 
# * Author:     Rodrigo Andrés Morales Mendoza.
                JuanJosé Zamudio
# * 28. dezember 2016.

#### ---- Proyecto de comunicación Banxico ---- ####
#### -----. Prueba de similiaridad entre .------ ####
###### ------- Minuta y Corredurías ---------- #####

@author: M15169
"""
# Nota: código subido a H: el 11ene17 por JJ,
#        y posteriormente modificado por RM

#Resta:
# 0. Checar por qué no corre (Inplot, check return plots)
# 1. agregar stemming (spanish) a código con nltk
# 2. Correr en paralelo
# 3. automatizar [minutas number] en "todo(,,)"
# 4. x-axis (fechas), pendiente.

print(chr(27) + "[2J")  #to clear terminal
# Get packages:
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import sklearn as sl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import pandas as pd
from stop_words import get_stop_words
#%matplotlib inline #this only works for jupyter notebook

# Define important functions:
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

def quitar_acentos(doc):
    a = doc.replace("á","a")
    e = a.replace("é","e")
    i = e.replace("í","i")
    o = i.replace("ó","o")
    u = o.replace("ú","u")
    n = u.replace("ñ","gn")
    s = n.replace('\n',' ')
    return s

def tokenizacion(doc): 
    return doc.split(" ")

def lista_flatten(lista):
    return [word for sentence in lista for word in sentence]

def palabras_unicas(terms):
    return list(set(terms))

def eliminar_puntuacion(term):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    return term.translate(replace_punctuation)

import re
def eliminar_numeros(term):
    return re.sub(r'\d+','',term)
def eliminar_espacios(term):
    return term.strip() 
def a_minusculas(term):
    return term.lower()
def eliminar_stopwords(term, stopwords):
    return list(set(term) - set(stopwords))
def frequency_count(term, words_in_docs):
    return [doc.count(term) for doc in words_in_docs]
def inverse_document_frequency(term, collection):
    bidf = [term in doc for doc in collection]
    return sum(bidf)


#Importar Pdfs y hacerlos una colección de docs
#Número de minuta
def import_txt(n,path):

    docs_collection=[]
    analistas=[]
    #analistas=["minuta"]
        
    #Minuta
    minuta= convert_pdf_to_txt(path+"minuta"+ str(n) + ".pdf")
    docs_collection.append(minuta)
        
    #Obtener corredurías, ir actualizando esta lista
    corr=["Banorte","BMX","Credit Suisse","Deutsche Bank","Barclays"]    
    
    for c in corr:
        try:
            docs_collection.append(convert_pdf_to_txt("l_" + str(c)+"_" + str(n) + "_post minuta.pdf"))
            analistas.append(c)
        except:
            continue
    
    return analistas,docs_collection

#Para obtener la Matriz tf-idf
def tfidf_mat(docs_collection):
    
    #Importar stop words
    my_stop_words = get_stop_words('spanish')
    my_stop_words.extend(['minuto','minutos'])
    
    terms = [tokenizacion(doc) for doc in docs_collection]
    
    tfidf_setting = TfidfVectorizer(docs_collection,encoding='utf-8', strip_accents='unicode',
                                analyzer='word', lowercase = True, stop_words = my_stop_words,
                                use_idf = True, min_df=1, norm='l2')
    
    #Sparse matrix de tfidf
    tfidf_matrix = tfidf_setting.fit_transform(docs_collection)
    
    return tfidf_matrix

#Para obtener las distancias de cada correduría respecto a su minuta
#Output es un diccionario de diccionario donde la sllaves son métricas y analistas

def get_distances(matrix,analistas):
    n=matrix.shape[0]
    m=len(analistas)
    dist= [ 'cosine', 'euclidean', 'manhattan']
    
    distances = {k: {a:() for a in range(1,m+1)} for k in dist}
    #distances={a:{k:[] for k in dist} for a in range(1,m+1)}
    
   
    #for a in analistas:
    for d in dist:
        for x in xrange(1,n):
                y=pairwise_distances(matrix[0],matrix[x], metric=d)
                distances[d][x]=y[0][0]
    
    #Renombrar con analistas
    for d in dist:
        for y in range(1,m+1):
            distances[d][analistas[y-1]] = distances[d].pop(y)
               
    return distances

def graficar(dic):
    n=len(dic)
    #Para el eje y (fechas/minutas)
    fechas=[x+1 for x in range(0,n)]
    #Obtener lista de métricas
    dist=dic[0].keys()
    
    #Crear lista de analistas que aparecen en alguno de los meses
    list_b=[]
    for p in range(0,n):
        
        a=dic[p]['cosine'].keys()
        list_b.extend(a)
        list_a=set(list_b)
    
    #Reorganizar diccionario para graficar mas facil
    dic_graf= {d: {a:[] for a in list_a} for d in dist}

    for p in range(0,n):
        for d in dist:
            for c in list_a:

                if c in dic[p][d]:
                    dic_graf[d][c].append(dic[p][d][c])
                else:
                    dic_graf[d][c].append(float('nan'))
    
    #Generar una gráfica por cada métrica
    for d in dist:
        colors = list("rgbcmyk")
        for k in dic_graf[d].values():
            plt.scatter(fechas,k,color=colors.pop())
            
        plt.legend(dic_graf[d].keys())
        plt.ylabel(d)
        plt.xlabel('Fechas')
            
        plt.show()   
    
    print "Done"

#Para correr todas las funciones. Poner rango de minutas.
#Output son las gráficas de las métricas que se pusieron en la función  get_distances()
def todo(n0,n1,path):
    graph_list=[]
    
    for x in range(n0,n1+1):
        print "Minuta",x
        analistas,docs=import_txt(x, path)
        mat= tfidf_mat(docs)
        
        #Generar una lista de diccionarios con todos los valores de métricas y analistas
        dist=get_distances(mat,analistas)
        graph_list.append(dist)
        
    graficar(graph_list)
    
    #return "Done"

#Ejemplo con minutas 33 y 34
#path = "C:\Users\M15169\Documents\DerRuediger\comunicacion_Bx\CorreduriasYminutas\"
path = "C:/Users/M15169/Documents/DerRuediger/comunicacion_Bx/CorreduriasYminutas/"
todo(33,34, path)





