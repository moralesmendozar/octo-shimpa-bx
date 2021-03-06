# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:01:02 2016

@author: M15169
"""

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

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
    
#txt1 = convert_pdf_to_txt("C:\pyd\pdfminer-master\samples\simple1.pdf")
txt1 = convert_pdf_to_txt("C:\pyd\correduria_001.pdf")
txt2 = convert_pdf_to_txt("C:\pyd\minuta47.pdf")

print txt1
print "--------------------------------------------------"
print "--------------------------------------------------"
print "--------------------------------------------------"
print txt2