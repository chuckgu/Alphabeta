import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Reshape,RepeatVector,Flatten
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU,Decoder_attention
from library.Model_ENC_DEC import ENC_DEC
from library.Load_data import prepare_data,load_data,load_dict
from library.Convolutional_Layer import Convolution1D,MaxPooling1D

#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 


n_epochs = 100

optimizer="Adam" #RMSprop,SGD,Adagrad,Adadelta,Adam
loss='nll_multiclass_3d'

snapshot_Freq=5
sample_Freq=1
val_Freq=1

n_sentence=9000
n_batch=128
n_maxlen=29 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words_x=10000 ## max numbers of words in dictionary
n_words_y=10000 ## max numbers of words in dictionary
dim_word=1024  ## dimention of word embedding 

n_u = dim_word
n_h = 1024 ## number of hidden nodes in encoder

n_d = 1024 ## number of hidden nodes in decoder
n_y = dim_word

stochastic=False
shared_emb=True
verbose=1

nb_filters = 256
filter_length = 3

####Load data

print 'Loading data...'

load_file='data/conv_real.pkl'
dic_file='data/conv_real.dict.pkl'

train, valid, test = load_data(load_file,n_words=n_words_x, valid_portion=0.02,
                               maxlen=n_maxlen)

print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

#print '<training data>' 
seq,seq_mask,targets,targets_mask=prepare_data(train[0],train[1],n_maxlen)

val,val_mask,val_targets,val_targets_mask=prepare_data(valid[0],valid[1],n_maxlen)

targets[:-1]=targets[1:]

targets_mask[:-1]=targets_mask[1:]

val_targets[:-1]=val_targets[1:]

val_targets_mask[:-1]=val_targets_mask[1:]

worddict = dict()

worddict = load_dict(dic_file)

####build model

print 'Initializing model...'

mode='tr'

model = ENC_DEC(n_u,n_h,n_d,n_y,
                n_epochs,n_batch,n_gen_maxlen,n_words_x,n_words_y,dim_word,
                snapshot_Freq,sample_Freq,val_Freq,shared_emb)

model.add(Convolution1D(input_dim=dim_word,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
'''
output_size=(((n_maxlen +1 - filter_length) / 1) + 1) / 2

model.add(Convolution1D(input_dim=nb_filters,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
'''
model.add(Flatten())
output_size = nb_filters * (((n_maxlen +1 - filter_length) / 1) + 1) / 2
#output_size = nb_filters * (((output_size- filter_length) / 1) + 1) / 2


model.add(FC_layer(output_size, n_h))
#model.add(Drop_out(0.25))
model.add(Activation('relu'))
model.add(RepeatVector(1))

model.add(Decoder_attention(n_h,n_d,n_y))
model.compile()



filepath='save/conv_conv.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
    
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
        
'''
    i=20
    for j in range(i):
        k=np.random.randint(1,n_sentence)
        a=j+1
        print('\nsample %i >>>>'%a)
        prob,estimate=sampling(k,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words)
 


def sampling(i,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words):
    test=seq[:,i]
    test_mask=seq_mask[:,i]
    
    truth=targets[:,i]
    
    guess = model.gen_sample(test,test_mask,stochastic)
    
    print 'Input: ',' '.join(input.sequences_to_text(test))
    
    print 'Truth: ',' '.join(output.sequences_to_text(truth))
    
    prob=np.asarray(guess[0],dtype=np.float)
    
    estimate=guess[1]
    
    print 'Sample: ',' '.join(output.sequences_to_text(estimate))
    
    return prob,estimate 
'''


    