import numpy as np
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Flatten_3d,Flatten
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from library.Model_3d import NN_Model
from library.Load_image import load_list

from library.Convolutional_Layer import Convolution3D, MaxPooling3D
import library.external.np_utils as np_utils

batch_size = 16
nb_classes = 101
nb_epoch = 100

np.random.seed(1337)

nb_rnn=1024
# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 320
# number of convolutional filters to use
nb_filters = 64
# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3
nb_channel=3

nb_frames=3

print 'Loading list...'

data_list='data/ucf_data_all.pkl'


# the data, shuffled and split between tran and test sets
(X_train,Y_train), (X_valid,Y_valid),(X_test,Y_test) = load_list(data_list,batch_size)

print 'train sample_size: %i'%len(X_train)
print 'valid sample_size: %i'%len(X_valid)

'''
X_train = X_train.reshape(X_train.shape[0], 1, shapex, shapey)
X_test = X_test.reshape(X_test.shape[0], 1, shapex, shapey)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
'''
# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

#X_mask= np.ones((len(X_train),nb_filters)).astype("float32")
#X_valid_mask= np.ones((len(X_valid),nb_filters)).astype("float32")

print 'Building model...'
model = NN_Model(n_epochs=nb_epoch,n_batch=batch_size,val_Freq=1,snapshot=1)
#model.add(Convolution3D(nb_filters,nb_frames, nb_channel , nb_conv, nb_conv))
model.add(Convolution3D(64,1, nb_channel , nb_conv, nb_conv,padding=(1,1,0)))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 1),padding=(0,0,1)))

model.add(Convolution3D(128,nb_frames, 64 , nb_conv, nb_conv,padding=(1,1,1)))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 2)))

model.add(Convolution3D(256,nb_frames, 128 , nb_conv, nb_conv,padding=(1,1,1)))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 1),padding=(0,0,1)))


model.add(Convolution3D(256,nb_frames, 256 , nb_conv, nb_conv,padding=(1,1,1)))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 2)))

model.add(Convolution3D(256,nb_frames, 256 , nb_conv, nb_conv,padding=(0,0,1)))
model.add(Activation('relu'))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 2),padding=(1,1,0)))

#model.add(FC_layer(nb_filters * (nb_frame_all / nb_pool) * (shapex / nb_pool) * (shapey / nb_pool), 128))

model.add(Flatten(dim=2))
model.add(FC_layer(4608, 2048))
# model.add(Activation('relu'))
model.add(Drop_out(0.5))
model.add(FC_layer(2048, nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='SGD',loss='nll_multiclass',mask=False)
model.load('temp/Video_pretrained.pkl')
model.train(X_train, None , Y_train, X_valid, None, Y_valid)
#score = model.evaluate(X_test, Y_test,batch_size=batch_size, show_accuracy=True, verbose=1)
##print('Test score:', score[0])
#print('Test accuracy:', score[1])

