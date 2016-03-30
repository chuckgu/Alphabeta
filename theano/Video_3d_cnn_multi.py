import numpy as np

from library.Load_image import load_list
from library.Utils import Progbar,ndim_tensor,make_batches,slice_X
import library.Callbacks as cbks
import library.external.np_utils as np_utils
from library.Load_image import load_image
import warnings
import multiprocessing
from datetime import datetime
import time,os


batch_size = 16
nb_classes = 101
nb_epoch = 2
nb_frame_all=16
nb_channel=3
NUM_GPU=4
share_flag=False

nb_rnn=512
# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 320
# number of convolutional filters to use
nb_filters = 16
# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3

nb_frames=3


def train_model(gpu_id, data_queue, model_queue, num_epoch, num_batch,valid_data_queue,num_val_batch): 

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(gpu_id)

    import theano
    from library.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Flatten_3d,Flatten
    from library.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
    from library.Model_3d import NN_Model
    from library.Convolutional_Layer import Convolution3D, MaxPooling3D

    print 'Building model...',gpu_id
    model = NN_Model(n_epochs=nb_epoch,n_batch=batch_size,val_Freq=1)
    #model.add(Convolution3D(nb_filters,nb_frames, nb_channel , nb_conv, nb_conv))
    model.add(Convolution3D(nb_filters,nb_frames, nb_channel , nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling3D((nb_pool, nb_pool, nb_pool)))
    model.add(Drop_out(0.25))
    model.add(Convolution3D(nb_filters,nb_frames, nb_filters/2 , nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling3D((nb_pool, nb_pool, nb_pool)))
    model.add(Drop_out(0.25))
   
    '''
    model.add(Flatten_3d(dim=3))
    #model.add(FC_layer(nb_filters * (nb_frame_all / nb_pool) * (shapex / nb_pool) * (shapey / nb_pool), 128))
    
    model.add(GRU(302736,nb_rnn,return_seq=False))
    model.add(Drop_out())
    
    model.add(FC_layer(nb_rnn, nb_classes))
    '''
    model.add(Flatten_3d(dim=2))
    model.add(FC_layer(434304, 128))
    model.add(Activation('relu'))
    model.add(Drop_out(0.5))
    model.add(FC_layer(128, nb_classes))

    model.add(Activation('softmax'))
    
    model.compile(optimizer='Adam',loss='nll_multiclass',mask=False)
    #model.train(X_train, X_mask , Y_train, X_valid, X_valid_mask, Y_valid)
 
    # train the model
    best_loss=np.inf
    best_save='_'.join((gpu_id,datetime.now().strftime('%Y_%m_%d_%H_%M_%S.h5')))
    #nb_data_queue=len(data_queue)

    for epoch in range(nb_epoch):

	train_losses=[]
        for batch in range(num_batch):
            data=data_queue.get()
            loss=model.train_on_batch(load_image(data[0]), None , data[2])

            if np.isnan(loss[0]) or np.isinf(loss[0]):
                raise ValueError('NaN detected')
	    train_losses.append(loss[0])

	    print gpu_id,'epoch',epoch+1,'batch',batch+1,'loss',loss[0],'accuracy',loss[1]

	    if share_flag:
		    # after a batch a data, synchronize the model
		    #model_weight=[layer.params for layer in model.layers]
		    model_weight=model._get_weights()
		    # we need to send NUM_GPU-1 copies out
		    for i in range(1,NUM_GPU):
		    	model_queue[gpu_id].put(model_weight)

		    
		    for k in model_queue:
		        if k==gpu_id:
		            continue
		        # obtain the model from other GPU
		        weight=model_queue[k].get()
		
		        # sum it
		        for l,w in enumerate(weight):
		            #model_weight[l]=[w1+w2 for w1,w2 in zip(model_weight[l],w)]
			    model_weight[l]=model_weight[l]+w
			     

		    # average it
		    for l,w in enumerate(model_weight):
			#model_weight[l]=[d/NUM_GPU for d in w]
			model_weight[l]=w/NUM_GPU
		    model._set_weights(model_weight)

	valid_losses=[]
	valid_accuracy=[]
	for batch in range(num_val_batch):
            data=valid_data_queue.get()
            loss=model.test_on_batch(load_image(data[0]), data[1] , data[2])

            if np.isnan(loss[0]) or np.isinf(loss[0]):
                raise ValueError('NaN detected')
	    train_losses.append(loss[0])

	print 'validation:',gpu_id,'loss',np.average(valid_losses),'accuracy',np.average(valid_accuracy)


	    
        # after each epoch, try to save the current best model
        if best_loss>np.average(valid_losses):
            model.save()
            best_loss=np.average(valid_losses)

   
if __name__=='__main__':

	print 'Loading list...'

	data_list='data/ucf_data_all.pkl'


	# the data, shuffled and split between tran and test sets
	(X_train,Y_train), (X_valid,Y_valid),(X_test,Y_test) = load_list(data_list,batch_size)

	X_mask= np.ones((len(X_train),nb_frame_all)).astype("float32")
	X_valid_mask= np.ones((len(X_valid),nb_frame_all)).astype("float32")
	
	X_train=np.asarray(X_train)
	Y_train=np.asarray(Y_train)

	X_valid=np.asarray(X_valid)
	Y_valid=np.asarray(Y_valid)
	
	ins=[X_train,X_mask,Y_train]
        val_ins=[X_valid,X_valid_mask,Y_valid]
	
	nb_train_sample=len(X_train)
	nb_val_sample=len(X_valid)

	index_array = np.arange(nb_train_sample)
	val_index_array = np.arange(nb_val_sample)
	np.random.shuffle(index_array)
	np.random.shuffle(val_index_array)
	
	batches = make_batches(nb_train_sample, batch_size)
	val_batches = make_batches(nb_val_sample, batch_size)

	num_batch=nb_train_sample/batch_size/NUM_GPU
	num_val_batch=nb_val_sample/batch_size/NUM_GPU

	gpu_list=['gpu{}'.format(i) for i in range(NUM_GPU)]
	print 'gpu list:',gpu_list


	# for send the data
	# train data
	manager = multiprocessing.Manager()
	data_queue=manager.Queue(100000)
	print 'train data allocation..'
	for epoch in range(nb_epoch):

	    for batch_index, (batch_start, batch_end) in enumerate(batches):

		batch_ids = index_array[batch_start:batch_end]
		ins_batch = slice_X(ins, batch_ids)
				
		data_queue.put(ins_batch)

	print 'train sample_size: %i'%nb_train_sample
	print 'train batch_size: %i'%num_batch

	# valid data
	manager3 = multiprocessing.Manager()
	valid_data_queue=manager3.Queue(100000)
	print 'valid data allocation..'

	for batch_index, (batch_start, batch_end) in enumerate(val_batches):

	    batch_ids = val_index_array[batch_start:batch_end]
	    ins_batch = slice_X(val_ins, batch_ids)
				
	    valid_data_queue.put(ins_batch)

	print 'valid sample_size: %i'%nb_val_sample
	print 'valid batch_size: %i'%num_val_batch


	manager2 = multiprocessing.Manager()
	# for synchronize the model, we create a queue for each model
	model_queue={gpu_id:manager2.Queue(10) for gpu_id in gpu_list}
	
	threads=[multiprocessing.Process(target=train_model, args=(gpu_id, data_queue, model_queue,nb_epoch,num_batch,valid_data_queue,num_val_batch)) for gpu_id in gpu_list]


	for thread in threads:
	    thread.start()

	for thread in threads:
	    thread.join()

