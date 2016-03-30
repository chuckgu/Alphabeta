import numpy as np
import matplotlib.pyplot as plt
import os
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Attention
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from library.Model_QA import NN_Model
from library.Load_data import load_data_with_image,prepare_full_data_keras_img
from library.Utils import Progbar,normalize
from sklearn.metrics import accuracy_score
from library.Modified_Layers import GRU2,GRU3,LSTM2,BiDirectionGRU2,Attention2,Attention3


#theano.config.exception_verbosity='high'

#theano.config.optimizer='None'

n_epochs = 100
optimizer="Adam"
loss='nll_multiclass'
#RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=10
sample_Freq=0
val_Freq=1


n_sentence=100000
n_batch=128
n_maxlen=30 ##max length of sentences in tokenizing
n_gen_maxlen=200 ## max length of generated sentences
n_words=12047 ## max number of words in dictionary
dim_word=1024# dimention of word embedding

n_u = dim_word
n_h = 1024 ## number of hidden nodes in encoder


stochastic=False
use_dropout=True
verbose=1

L1_reg=0
L2_reg=0

print 'Loading data...'

load_train='data/train_vgg.pkl'
load_test='data/test_vgg.pkl'
#load_img='data/train_vgg.pkl'

train, valid, test = load_data_with_image(load_train,load_test,n_words=n_words, valid_portion=0.000,
                               maxlen=n_maxlen,max_lable=None)


n_y = np.max((np.max(train[1]),np.max(test[1]))) + 1

print 'number of classes: %i'%n_y
print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

####build model
print 'Initializing model...'

mode='tr'

model = NN_Model(n_epochs=n_epochs,n_batch=n_batch,snapshot=snapshot_Freq,
            sample_Freq=sample_Freq,val_Freq=val_Freq,L1_reg=L1_reg,L2_reg=L2_reg)
model.add(Embedding(n_words,dim_word))
model.add(Drop_out(0.25))
model.add(GRU3(n_u,n_h,return_seq=False))
model.add(Drop_out())
#model.add(Attention3(n_h,n_h))
#model.add(Drop_out())
model.add(FC_layer(n_h,n_y))
model.add(Activation('softmax'))
model.compile(optimizer=optimizer,loss=loss)



filepath='save/qna.pkl'

if mode=='tr':
    #if os.path.isfile(filepath): model.load(filepath)
    print '<training data>'
    seq,seq_mask,targets,train_img=prepare_full_data_keras_img(train[0],train[1],train[2],n_maxlen)
    print '<validation data>'
    val,val_mask,val_targets,val_img=prepare_full_data_keras_img(test[0],test[1],test[2],n_maxlen)

    #train_img=normalize(train_img)
    #val_img=normalize(val_img)

    model.train(seq,seq_mask,targets,train_img,val,val_mask,val_targets,val_img,verbose)
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

    test,tets_mask,test_targets,test_img=prepare_full_data_keras_img(test[0],test[1],test[2],n_maxlen)
    test_img=normalize(test_img)

    print 'Testing model...'
    score = model.evaluate(test, tets_mask,test_targets,test_img, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])