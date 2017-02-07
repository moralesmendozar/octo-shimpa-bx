# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:10:38 2017

@author: M15169
"""

#quitar puntuación
import string 

string.punctuation


#ttt = convert_pdf_to_txt("C:\pyd\minuta47.pdf")
ttt = convert_pdf_to_txt("C:\pyd\minuta33.pdf")
ddd = quitar_acentos(ttt)
ddd = a_minusculas(ddd)
ddd = eliminar_puntuacion(ddd)
termsddd = tokenizacion(ddd)


#termsii = [SnowballStemmer("spanish").stem(term) for term in termsddd]

nn = len(termsddd)
a = 0
b = nn
givenrange = range(a,b)
for jj in givenrange:
    stemmed = SnowballStemmer("spanish").stem(termsddd[jj])    
    print jj
    print stemmed
    

num = 2089
termw = termsddd[num]
print termw
SnowballStemmer("spanish").stem(termw)

num = 6358
termw = termsddd[num]
print termw






detectednumbers = [253, 255, 2220, 2288, 2302, 2302, 2325, 2327, 2328,2418, 2422, 2430]
for k in detectednumbers:
    print termsddd[k]