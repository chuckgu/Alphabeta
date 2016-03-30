import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU,Decoder_attention
from library.Model_ENC_DEC import ENC_DEC
from library.Load_data import prepare_data,load_data,load_dict
from nltk.tokenize import sent_tokenize,word_tokenize
from library.Utils import seq_to_text,text_to_tuple


from konlpy.tag import Twitter
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 


n_epochs = 1000



optimizer="Adam" #RMSprop,SGD,Adagrad,Adadelta,Adam
loss='nll_multiclass_3d'

snapshot_Freq=5
sample_Freq=1
val_Freq=1

n_sentence=9000
n_batch=16
n_maxlen=29 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words_x=265 ## max numbers of words in dictionary+2 301
n_words_y=135 ## max numbers of words in dictionary+2 138
dim_word=50  ## dimention of word embedding 

n_u = dim_word
n_h = 512 ## number of hidden nodes in encoder

n_d = 512 ## number of hidden nodes in decoder
n_y = dim_word

stochastic=True
shared_emb=False
verbose=1

####Load data

print 'Loading data...'

load_file='data/xproject_josa.pkl'
dic_file_x='data/xproject_josa.dict1.pkl'
dic_file_y='data/xproject_josa.dict2.pkl'

train, valid, test = load_data(load_file,n_words=n_words_x, valid_portion=0.2,
                               maxlen=n_maxlen)

print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

#print '<training data>' 
seq,seq_mask,targets,targets_mask=prepare_data(train[0],train[1],n_maxlen)

val,val_mask,val_targets,val_targets_mask=prepare_data(valid[0],valid[1],n_maxlen)

#targets[:-1]=targets[1:]

#targets_mask[:-1]=targets_mask[1:]

#val_targets[:-1]=val_targets[1:]

#val_targets_mask[:-1]=val_targets_mask[1:]

worddict_x = dict()

worddict_x = load_dict(dic_file_x)

worddict_y = dict()

worddict_y = load_dict(dic_file_y)

worddict=[worddict_x,worddict_y]

####build model

print 'Initializing model...'

mode='te'

model = ENC_DEC(n_u,n_h,n_d,n_y,
                n_epochs,n_batch,n_gen_maxlen,n_words_x,n_words_y,dim_word,
                snapshot_Freq,sample_Freq,val_Freq,shared_emb)

model.add(GRU(n_u,n_h))
#model.add(Drop_out())
model.add(Decoder(n_h,n_d,n_y))
model.compile()



filepath='save/xproject_josa.pkl'

if mode=='tr':
    #if os.path.isfile(filepath): model.load(filepath)
    
    train_set=[seq,seq_mask,targets,targets_mask]
    val_set=[val,val_mask,val_targets,val_targets_mask]
    
    model.train(train_set,val_set,worddict,verbose)
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
        seqs = [s if s<n_words_x else 1 for s in seqs]
        mask_set_x=np.ones((len(seqs))).astype('float32')
        #print seqs
        
        test=seqs
        mask=mask_set_x
        
        ins_gen=[test,mask]
        
        res=model.generate(ins_gen,worddict,with_truth=False,stochastic=stochastic)[1]
        
        #print res
        print "Sulim(Wife): "+text_to_tuple(seq_to_text(np.asarray(res),worddict[-1]))
     
    

        


    