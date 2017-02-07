# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:59:52 2017
# * Banco de México
# * Gerencia de Investigación del Sector Real 
# * Author:     Rodrigo Andrés Morales Mendoza.
                JuanJosé Zamudio
# * 11-13. januar 2016.

#### ---- Proyecto de comunicación Banxico ---- ####
#### -----. Prueba de similiaridad entre .------ ####
###### ------- Minuta y Corredurías ---------- #####

@author: M15169
"""
# Nota: código subido a H: el 11ene17 por JJ,
#        y posteriormente modificado por RM

#últimas modificaciones.
# check :)  0. Checar por qué no corre (Inplot, check return plots) -- no tenía los pdfs en la carpeta
# check ;)  1. agregar stemming (spanish) a código con nltk
# check :O  2. Correr en paralelo... por alguna razón corre más rápido en simple que en paralelo... makes no sense
# check ;) 3. automatizar [minutas number] en "todo(,,)"

#Resta:
#lo único que dejo pendiente...
#PENDIENTE
# 4. x-axis (fechas), pendiente.
# 5. Se puede quitar números, después de quitar acentos en las minutas, etc.

print(chr(27) + "[2J")  #to clear terminal
# Get packages:
import time
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
from pylab import *
#quitar puntuación
import string 
string.punctuation
#para el stemming:
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
stemmer = PorterStemmer()
#Para hacer multicore-processing  https://docs.python.org/3/library/multiprocessing.html
from multiprocessing import Pool
#from multiprocessing import Process
#%matplotlib inline #this only works for jupyter notebook

# Define important functions:

############### Funciones para word processing...
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
    
def getTxt(path):
    try:
        txt = convert_pdf_to_txt(path)
        txt = quitar_acentos(txt)
        txt = a_minusculas(txt)
        txt = eliminar_puntuacion(txt)
        txt = doStemmOnDi(txt)
        return txt
    except:
        return -1

def quitar_acentos(doc):
    a = doc.replace("á","a")
    a = a.replace("Á","A")
    e = a.replace("é","e")
    e = e.replace("É","E")
    i = e.replace("í","i")
    i = i.replace("Í","I")
    o = i.replace("ó","o")
    o = o.replace("Ó","O")
    u = o.replace("ú","u")
    u = u.replace("Ú","U")
    n = u.replace("ñ","gn")
    n = n.replace('Ñ','N')
    s = n.replace('\n',' ')
    # y quitamos los símbolos que nos han dado problemas:
    s = s.replace('“',' ')
    s = s.replace('”',' ')
    s = s.replace('°',' ')
    s = s.replace('←',' ')
    s = s.replace('',' ')
    s = s.replace('',' ')
    s = s.replace('…',' ')
    s = s.replace('’',' ')
    s = s.replace('–',' ')
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
    #analistas=["minuta"]
    pool = Pool(processes=4)    
    #proc = []  #Contains the processes to be run on parallel
    #Minuta
    pathin = path+"minuta"+ str(n) + ".pdf"
    minuta = convert_pdf_to_txt(pathin) #sin parallell
    #stuff = pool.apply_async(*make_applicable(convert_pdf_to_txt,pathin)) #con parallell
    #minuta = stuff.get(timeout=15) #con parallell
    minuta = quitar_acentos(minuta)
    minuta = a_minusculas(minuta)
    minuta = eliminar_puntuacion(minuta)
    docs_collection.append(minuta)
    analistas =[]
    
    # Boletin
    pathin = "boletin"+ str(n) + ".pdf"
    boletin = convert_pdf_to_txt(pathin) #sin parallell
    #stuff = pool.apply_async(*make_applicable(convert_pdf_to_txt,pathin)) #con parallell
    #minuta = stuff.get(timeout=15) #con parallell
    boletin = quitar_acentos(boletin)
    boletin = a_minusculas(boletin)
    boletin = eliminar_puntuacion(boletin)
    docs_collection.append(boletin)
    #analistas.append(pathin)
    analistas.append("Boletin "+ str(n))
    
    #Obtener corredurías, ir actualizando esta lista
    corr=["Banorte","BMX","Credit Suisse","Deutsche Bank","Barclays"]
    pathins = [ "l_" + str(c)+"_" + str(n) + "_post minuta.pdf" for c in corr]
#    print 'pathins = ', pathins   
    resultsTxt = [getTxt(pathein) for pathein in pathins] #sin parallell
    #results = [pool.apply_async(*make_applicable(getTxt,pathein)) for pathein in pathins] #con parallell
    #resultsTxt = [result.get(timeout=10) for result in results]#con parallell
    counter = 0
    
    analistasUnused=[]
    for resultst in resultsTxt:
        if resultst == -1:  #esto quiere decir que no hay correduría
            ########## aquí se puede agregar algo que indique empty correduría
            #analistasUnused.append(pathins[counter])
            analistasUnused.append(corr[counter])
            ########## aquí se puede agregar algo que indique empty correduría
        else:
            docs_collection.append(resultst)
            #analistas.append(pathins[counter])
            analistas.append(corr[counter])            
        counter = counter + 1
    #results = getTxt(path)
#  for fn in fns:
#    p = Process(target=fn)
#    p.start()
#    proc.append(p)
#  for p in proc:
#    p.join()
    
#    for c in corr:
#        try:        
#            pathin = "l_" + str(c)+"_" + str(n) + "_post minuta.pdf"
#            
#            stuff = pool.apply_async(*make_applicable(convert_pdf_to_txt,pathin))
#            correduria = stuff.get(timeout=15)
#            #correduria = convert_pdf_to_txt(pathin)
#            correduria = quitar_acentos(correduria)
#            correduria = a_minusculas(correduria)
#            correduria = eliminar_puntuacion(correduria)
#            docs_collection.append(correduria)
#            analistas.append(c)
#        except:
#            ########## aquí se puede agregar algo que indique empty correduría
#            continue
    
    return analistas,docs_collection

#Para la stemmización
def doStemmOnDi(di):
    termsi = tokenizacion(di)
    termsii = [SnowballStemmer("spanish").stem(term) for term in termsi]
    termsiii = [ termsii[k].encode("utf-8") for k in range(len(termsii)) ]
    dii = ' '.join(termsiii)
    return dii

def doStemm(docs):
# this function receives docs, which contains documents
#    and returns docs2, which contains documents with the stemmed words in Spanisch
#    print 'docs[0] :'
#    print docs[0]
#    print '--------------------------------------------'
#    print 'len(docs) :'
#    print len(docs)
#    print '--------------------------------------------'
    docs2 = []    
    for i in range(len(docs)):
        print i
        di = docs[i]
        dii = doStemmOnDi(di)
        docs2.append(dii)
    return docs2

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

############### fin de funciones para word processing... (faltan las de distances)

#Para obtener las distancias de cada correduría respecto a su minuta
#Output es un diccionario de diccionario donde la sllaves son métricas y analistas
def get_distances(matrix,analistas):
    n=matrix.shape[0]
    m=len(analistas)
    dist= ['euclidean', 'manhattan', 'cosine']
    distances = {k: {a:() for a in range(1,m+1)} for k in dist}
    #print 'distances = ', distances
    #distances={a:{k:[] for k in dist} for a in range(1,m+1)}   
    #for a in analistas:
    for d in dist:
        for x in range(1,n):
                y=pairwise_distances(matrix[0],matrix[x], metric=d)
                distances[d][x]=y[0][0]    
    #Renombrar con analistas
    for d in dist:
        for y in range(1,m+1):
            distances[d][analistas[y-1]] = distances[d].pop(y)
    
    return distances


def graficar(dic):
    plotslist = []
    #print 'dic = ', dic
    n=len(dic)  #es el número de minutas a estudiar
    #print 'n = ', n
    #Para el eje y (fechas/minutas)
    fechas=[x+1 for x in range(0,n)]  #es de 1 al número de minutas a estudiar
    #Obtener lista de métricas
    dist=dic[0].keys()
    
    #Crear lista de analistas que aparecen en alguno de los meses
    list_b=[]
    for p in range(0,n):
        
        a=dic[p]['cosine'].keys()
        list_b.extend(a)
        list_a=set(list_b)
    print 'list_a = ', list_a
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
    plotcounter = 0
    #print 'dist = ', dist
    for d in dist:
        #colors = list("rgbcmykkk")
        colors = [(0.2,0.2,0.4),(0.5,0.5,0.5),(0.5,0.5,1),(0.5,1,0.5),(0.5,1,1),(1,0.5,0.5),(1,0.5,1),(1,1,0.5),(0.8,0.8,0.8)]
        # Para hacer plots independientes
        fig = figure()        
        #plt.figure(plotcounter)
        print ' dic_graf[d].values() = ', dic_graf[d].values()
        countcolor = 0
        for k in dic_graf[d].values():
            #_ = scatter(fechas,k,color=colors[countcolor])
            _ = scatter(fechas,k,color=colors.pop())
            countcolor = countcolor + 1
        legend(dic_graf[d].keys())
        ylabel(d)
        xlabel('Fechas')
        
        plotslist.append(fig)
        plotcounter = plotcounter + 1
        #plt.show()   
    return plotslist
    print "Done"
    


#Para correr todas las funciones. Poner rango de minutas.
#Output son las gráficas de las métricas que se pusieron en la función  get_distances()
def todo(rangeOfMinutas,path):
    graph_list=[]
    for x in rangeOfMinutas:#range(n0,n1+1):
        print "Minuta",x
        analistas,docs=import_txt(x, path)
        
        docs = doStemm(docs) #ya está el Stemming en getText  (en import_txt)
#==============================================================================
#         minuta = "Si inflación sube, se subirá la tasa en el futuro"
#         c1 = "Subirían tasa el próximo mes"
#         c2 = "La tasa no sube si inflación no sube"
#         docs = [minuta,c1,c2]
#==============================================================================
        mat= tfidf_mat(docs)
        #print mat
        #Generar una lista de diccionarios con todos los valores de métricas y analistas
        
        dist=get_distances(mat,analistas)
        graph_list.append(dist)
#    print 'graph_list:'
#    print graph_list
#    print 'done printing praph_list'
    graphs = graficar(graph_list)
#    print('printing graphs.... ')
#    print graphs
#    print('printed graphs.')
#    print('len(graphs) = ')
#    print len(graphs)
#    print('graphs[0].... ')
#    print graphs[0]
    return graphs
#    #return "Done"

#Ejemplo con minutas 33 y 34
#path = "C:\Users\M15169\Documents\DerRuediger\comunicacion_Bx\CorreduriasYminutas\"

if __name__ == '__main__':
    path = "C:/Users/M15169/Documents/DerRuediger/comunicacion_Bx/CorreduriasYminutas/"
    tic = time.clock()
    #pool = Pool(processes = 4)
    #pool.map(todo, [33,34], path)
    todo([33,34], path)
    #graphs = todo([33,34], path)
    toc = time.clock()
    print '------------------------------------------'
    print 'Total running time was =  ', toc-tic
    print '------------------------------------------'
    #pool = Pool(processes=4)
    #graphs = pool.apply(todo, ( [33,34], path, ) )
    #graphs = pool.apply_async(todo, ( [33,34], path, ) )
    #print(graphs.get(timeout=1))





