# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:35:54 2016

@author: chuckgu
"""

import json,os
from nltk.tokenize import sent_tokenize,word_tokenize
from konlpy.tag import Twitter
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

twitter=Twitter()

txt=[]

checklist=['Exclamation','Alpha','URL','Punctuation','Foreign','Unknown','Hashtag','ScreenName','Josa']

'''
currdir = os.getcwd()
os.chdir('%s/' % currdir)
print currdir

with open("text8", 'r') as f:
    for line in f:
        sentences.append(line[:100])
        
print sentences 
'''      
with open("/home/chuckgu/Desktop/project/preprocessing/x-project/word2vec/namuwiki160229/namuwiki_20160229.json") as json_file:
    json_data = json.load(json_file)

for i,j in enumerate(json_data): 
    print i
    
    sentences=sent_tokenize(j["text"])
    
    if len(sentences)>5:
        for line in sentences:
            line=line.decode('utf-8')
            #txt.append(' '.join(twitter.morphs(line)))
            txt.extend([s[0]for s in twitter.pos(line,norm=True)  if s[1] not in checklist])
            
    if i==120000: break
    
#np.savetxt("namu.txt",txt,fmt='%s') 
import cPickle as pkl
f = open('namu_wo_josa.pkl', 'wb')
pkl.dump(txt, f, -1)
f.close()
print 'saved'