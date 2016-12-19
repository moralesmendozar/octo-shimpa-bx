# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:01:02 2016

@author: M15169
"""
# Clear stuff:
print(chr(27) + "[2J")  #to clear terminal
print 'Inicia el programa pdfminer_and_tfidf_test01.py'


# Get packages:
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

import sklearn as sl
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words



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
# y algunas otras funciones importantes
############# Quitar acentos, después se vuelve más complicado¡!
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
#------------------------------------------------------------------


###### Obtenemos los textos de los pdfs correspondientes:  
#txt1 = convert_pdf_to_txt("C:\pyd\pdfminer-master\samples\simple1.pdf")
print 'get txt1 - correduría...'
txt1 = convert_pdf_to_txt("C:\pyd\correduria_001.pdf")
print 'get txt2 - minuta...'
txt2 = convert_pdf_to_txt("C:\pyd\minuta47.pdf")
#d1 = txt1.encode('utf-8')  #descifrar cómo hacer esto.
#print 'txt1: '
#print txt1
#print "--------------------------------------------------"
#print "--------------------------------------------------"
#print "--------------------------------------------------"
#print 'txt2: '
#print txt2
print 'Quitando acentos...'
d1 = quitar_acentos(txt1)
d2 = quitar_acentos(txt2)
#creamos nuestra collección de documentos para hacer diccs y esas cosas:
docs_collection = [d1, d2] 
#obtenemos los términos independientes:
print 'begin tokenization...'
terms = [tokenizacion(doc) for doc in docs_collection]
#print 'terms'
#print terms
#print '---------------------'


#stopwords come from: https://pypi.python.org/pypi/stop-words
# stopwords = [word.decode('utf-8') for word in stopwords.words('spanish')]
my_stop_words = get_stop_words('spanish')
#my_stop_words= ['el','la']  #se pueden definir las propias stopwords

### from stop_words import safe_get_stop_words
### stop_words = safe_get_stop_words('unsupported language')
print 'vectorizing with TfidfVectorizer()...'
tfidf_setting = TfidfVectorizer(encoding='utf-8', strip_accents='unicode',
                                analyzer='word', lowercase = True, stop_words = my_stop_words,
                                use_idf = True, min_df=0)
print 'getting tfidf_matrix() ...'                            
tfidf_matrix = tfidf_setting.fit_transform(docs_collection)
print '------------------------------------------'
print 'this is the tfidf_matrix: '
print ''
print tfidf_matrix
############ Checar qué onda con este código comentado:
for col in tfidf_matrix.nonzero()[1]:
    print col
    print terms[col], '-', tfidf_matrix[0,col]
    
q = 'La tasa de interes ha alivianado presiones inflacionarias luego de que la brecha de producto era muy alta'
#eliminar acentos
documento_prueba = quitar_acentos(q)
q_wo_accents = quitar_acentos(q) 


response = tfidf_setting.transform([q_wo_accents])
#print response

print "-------------------------------------------------------------?"

for col in response.nonzero()[1]:
    print terms[col], ' - ', response[0, col]
