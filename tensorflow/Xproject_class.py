import numpy as np
import matplotlib.pyplot as plt
import os
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from library.Model import NN_Model
from library.Load_data import load_data,prepare_full_data_keras,load_dict
from library.Utils import Progbar
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize,word_tokenize
from library.Utils import seq_to_text,text_to_tuple


from konlpy.tag import Twitter
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 

n_epochs = 1000
optimizer="Adam"
loss='nll_multiclass'
#RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=25
sample_Freq=0
val_Freq=1


n_sentence=100000
n_batch=128 
n_maxlen=30 ##max length of sentences in tokenizing
n_gen_maxlen=200 ## max length of generated sentences
n_words=569 ## max number of words in dictionary
dim_word=128# dimention of word embedding 

n_u = dim_word
n_h = 512 ## number of hidden nodes in encoder


stochastic=False
use_dropout=True
verbose=1

L1_reg=0
L2_reg=0.001

print 'Loading data...'

load_file='data/xproject_class.pkl'
dic_file_x='data/xproject_class.dict.pkl'

train, valid, test = load_data(load_file,n_words=n_words, valid_portion=0.2,
                               maxlen=n_maxlen,max_lable=None)
        
n_y = np.max((np.max(train[1]),np.max(valid[1]))) + 1

print 'number of classes: %i'%n_y
print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

worddict_x = dict()

worddict_x = load_dict(dic_file_x)

####build model
print 'Initializing model...'

mode='te'

model = NN_Model(n_epochs=n_epochs,n_batch=n_batch,snapshot=snapshot_Freq,
            sample_Freq=sample_Freq,val_Freq=val_Freq,L1_reg=L1_reg,L2_reg=L2_reg)
model.add(Embedding(n_words,dim_word))            
model.add(Drop_out(0.2))
model.add(GRU(n_u,n_h,return_seq=False))
model.add(FC_layer(n_h,n_y))
model.add(Activation('softmax'))
model.compile(optimizer=optimizer,loss=loss)



filepath='save/xproject_result.pkl'

if mode=='tr':
    #if os.path.isfile(filepath): model.load(filepath)
    print '<training data>'    
    seq,seq_mask,targets=prepare_full_data_keras(train[0],train[1],n_maxlen)
    print '<validation data>'
    val,val_mask,val_targets=prepare_full_data_keras(valid[0],valid[1],n_maxlen)

    model.train(seq,seq_mask,targets,val,val_mask,val_targets,verbose)
    model.save(filepath)
    
    ##draw error graph 
    plt.close('all')
    fig = plt.figure()
    ax3 = plt.subplot(111)   
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')    
    plt.savefig('error.png')
    
    
elif mode=='te':
    if os.path.isfile(filepath): model.load(filepath)
    else: 
        raise IOError('loading error...')
    
    checklist=['Exclamation','Alpha','URL']
    twitter=Twitter()

    
    while 1:
        choice=raw_input("Me: ")
        if choice in ["Q","q"]: break
        #print choice
        
        choice=choice.decode('utf-8')
        
        sen=' '.join([s[0]+'/'+s[1] for s in twitter.pos(choice,norm=True)  if s[1] not in checklist])
        

        words=(word_tokenize(sen.strip().lower()))
        #print ' '.join(words)
        seqs = [worddict_x[w] if w in worddict_x.keys() else 1 for w in words]
        seqs = [s if s<n_words else 1 for s in seqs]
        mask_set_x=np.ones((len(seqs))).astype('float32')
        res=model.predict(seqs,mask_set_x)
        
        #print res
        print "class: "+str(res)
