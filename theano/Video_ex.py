import numpy as np
from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Flatten_3d,Flatten
from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from library.Model import NN_Model
from library.Load_image import load_list
import shapes_3d
from library.Convolutional_Layer import Convolution3D, MaxPooling3D
import library.external.np_utils as np_utils


# Data Generation parameters
test_split = 0.2
dataset_size = 5000
patch_size = 32

(X_train, Y_train),(X_test, Y_test) = shapes_3d.load_data(test_split=test_split,
                                                          dataset_size=dataset_size,
                                                          patch_size=patch_size)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# CNN Training parameters
batch_size = 10
nb_classes = 2
nb_epoch = 100
nb_pool=3

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

print 'Building model...'
model = NN_Model(n_epochs=nb_epoch,n_batch=batch_size,val_Freq=1)


#model.add(Convolution3D(nb_filters,nb_frames, nb_channel , nb_conv, nb_conv))
model.add(Convolution3D(16, 7, 1 , 7, 7,padding=(1,1,1)))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 2)))
model.add(Drop_out(0.5))

model.add(Convolution3D(32, 3, 16 , 3, 3,padding=(1,1,1)))
model.add(MaxPooling3D(poolsize=(nb_pool, nb_pool, nb_pool),stride=(2, 2, 2)))


model.add(Flatten_3d(dim=2))
model.add(Drop_out(0.5))
model.add(FC_layer(6912, 128))
model.add(FC_layer(128, nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='RMSprop',loss='nll_multiclass',mask=False)
model.train(X_train, None , Y_train, X_test, None, Y_test)
#score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
##print('Test score:', score[0])
#print('Test accuracy:', score[1])

