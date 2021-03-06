import theano
import theano.tensor as T
import numpy as np
import logging
import os
import datetime
import cPickle as pickle
from Optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam
from Loss import nll_multiclass,categorical_crossentropy
import Callbacks as cbks
from Utils import Progbar,ndim_tensor,make_batches,slice_X
from Load_image import load_image

mode = theano.Mode(linker='cvm', optimizer='fast_run') #the runtime algo to execute the code is in c

class NN_Model(object):   
    def __init__(self,n_epochs=100,n_batch=128,snapshot=20,sample_Freq=1,val_Freq=1,L1_reg=0,L2_reg=0):
                
        self.n_batch=int(n_batch)
        self.n_epochs=n_epochs
        
        self.layers = []
        self.decoder=[]
        self.params=[]
        self.params_layer=[]
        self.errors=[]
        self.val_errors=[]
        
        self.snapshot=int(snapshot)
        self.sample_Freq=int(sample_Freq)
        self.val_Freq=int(val_Freq)
       
        self.L1_reg=L1_reg
        self.L2_reg=L2_reg    
        self.L1= 0
        self.L2_sqr= 0
        
        
    def add(self,layer): 
  
        self.layers.append(layer)
    
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.set_input()
            #self.layers[0].set_mask(self.x_mask)
  
        self.params+=layer.params
        self.params_layer.append(len(self.layers))
        self.L1 += layer.L1
        self.L2_sqr += layer.L2_sqr


    def set_params(self,**params):
        return
    
    def __getstate__(self):
        """ Return state sequence."""
        params = self.params  # parameters set in constructor
        weights = [p.get_value() for p in self.params]
        error=self.errors
        val_error=self.val_errors
        params_layer=self.params_layer
        state = (params, params_layer, weights,error,val_error)
        return state

    def _get_weights(self):
        return [p.get_value(borrow=False) for p in self.params]


    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params,params_layer, weights, error,val_error = state
        #self.set_params(**params)
        #self.ready()
        self._set_weights(weights)
        self.errors=error
        self.val_errors=val_error
        self.params_layer=params_layer

    def save(self, fpath='temp/', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)


        print("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """

        print("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()       

    def load_layers(self, path, layers):
        """ Load model parameters from path. """

        
        file = open(path, 'rb')
        params,params_layer, weights, error,val_error = pickle.load(file)

        i = iter(weights)

        for j,param in enumerate(self.params):
	    print("Loading weight %s ..." % j)
            param.set_value(i.next().get_value())
	    if j==5: break
	    
        file.close()     
        
    
    def get_output(self,train=False):
        
        return self.layers[-1].get_output(train)


    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim
                dtype=l.input.dtype
                self.layers[0].input = ndim_tensor(ndim,dtype)
                break

    def get_input(self,train):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)  
    
    def set_mask(self):
        ndim = self.layers[0].ndim
        self.layers[0].x_mask = ndim_tensor(ndim,'float32')
      
    def get_mask(self):
        if not hasattr(self.layers[0], 'x_mask'):
            self.set_mask()
        return self.layers[0].get_mask()
        

        
    def compile(self,optimizer='Adam',loss='nll_multiclass',mask=True):      

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)


        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        if loss=='nll_multiclass': self.y = T.vector(name = 'y', dtype = 'int32')
        else:self.y = T.zeros_like(self.y_train)
        
        if mask:
            self.X_train_mask = self.get_mask()
            self.X_test_mask = self.get_mask()
            
            if type(self.X_train) == list:
                train_ins = self.X_train + self.X_train_mask+[self.y]
                test_ins = self.X_test + self.X_test_mask + [self.y]
                predict_ins = self.X_test + self.X_test_mask
            else:
                train_ins = [self.X_train, self.X_train_mask, self.y]
                test_ins = [self.X_test, self.X_test_mask, self.y]
                predict_ins = [self.X_test, self.X_test_mask]
        else:
            if type(self.X_train) == list:
                train_ins = self.X_train +[self.y]
                test_ins = self.X_test  + [self.y]
                predict_ins = self.X_test 
            else:
                train_ins = [self.X_train, self.y]
                test_ins = [self.X_test, self.y]
                predict_ins = [self.X_test]  
                
        ### cost and updates    
 
        self.loss = eval(loss)


        train_loss=self.loss(self.y, self.y_train)
        test_loss=self.loss(self.y, self.y_test)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        
        if loss=='nll_multiclass':
            train_accuracy = T.mean(T.eq(self.y, T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(self.y, T.argmax(self.y_test, axis=-1)))   
        else:
            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

        
        cost = train_loss +self.L2_reg * self.L2_sqr
        
        self.optimizer=eval(optimizer)()
        updates=self.optimizer.get_updates(self.params,cost)    

        print 'Optimizer: '+optimizer   
        
        ### compile theano functions 
        
                          
        self._train_with_acc = theano.function(inputs = train_ins,
                                      outputs = [train_loss,train_accuracy],
                                      updates = updates,
                                      mode = mode,
                                      allow_input_downcast=True,
                                      on_unused_input='ignore')

        self._test_with_acc = theano.function(inputs = test_ins,
                                              outputs = [test_loss,test_accuracy],
                                              mode = mode,
                                              allow_input_downcast=True,
                                              on_unused_input='ignore') 
        
        self._predict = theano.function(inputs = predict_ins,
                                             outputs = self.y_test,
                                             mode = mode,
                                              allow_input_downcast=True,
                                              on_unused_input='ignore')
                                             

    

    def train(self,X_train,X_mask,Y_train,X_val,X_val_mask,Y_val,verbose=1,shuffle=True, show_accuracy=True):
        
        ### input data        
    
        train_set_x = np.asarray(X_train)
        train_set_y = np.asarray(Y_train)

        
        if X_mask is None:
            ins = [train_set_x, train_set_y]
        else:
            mask_set_x = np.asarray(X_mask)            
            ins = [train_set_x, mask_set_x, train_set_y]

        val_set_x = np.asarray(X_val)
        val_set_y = np.asarray(Y_val)   
          
        
        if X_val_mask is None:
            val_ins = [val_set_x, val_set_y]
        else:
            mask_val_set_x = np.asarray(X_val_mask)               
            val_ins = [val_set_x, mask_val_set_x, val_set_y]
          
        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'

        nb_train_sample = train_set_x.shape[0]
        index_array = np.arange(nb_train_sample)
        
        ### call back###
        history = cbks.History()
        
        callbacks = [history, cbks.BaseLogger()]

        callbacks = cbks.CallbackList(callbacks)
        

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
            metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        else:
            f = self._train
            out_labels = ['loss']
            metrics = ['loss', 'val_loss']        
        
        do_validation = True

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': self.n_batch,
            'nb_epoch': self.n_epochs,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        for epoch in range(self.n_epochs):
            callbacks.on_epoch_begin(epoch)
            
            if shuffle: np.random.shuffle(index_array)

            train_losses=[]
            #train_batch_sizes=[]

            #for idx in range(n_train_batches):
            batches = make_batches(nb_train_sample, self.n_batch)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                                     
                #batch_start,batch_stop=self.get_batch(idx,n_train)  
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                    if len(ins_batch)==3:
                        ins_batch=[load_image(ins_batch[0]),ins_batch[1],ins_batch[2]]
                    else: ins_batch=[load_image(ins_batch[0]),ins_batch[1]]

                except TypeError as err:
                    print('TypeError while preparing batch. \
                        If using HDF5 input data, pass shuffle="batch".\n')
                    raise
                   
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                        
                cost = f(*ins_batch)                    
                 
                if np.isnan(cost[0]) or np.isinf(cost[0]):
                    raise ValueError('NaN detected')
                                  
                train_losses.append(cost[0]) 
                #train_batch_sizes.append(self.get_batch_size(idx, n_train))  
                
                for l, o in zip(out_labels, cost):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

            epoch_logs = {} 
            # compute loss on validation set
            if np.mod(epoch+1,self.val_Freq)==0:

                val_outs = self._test_loop(self._test_with_acc, val_ins, batch_size=self.n_batch, verbose=1)
                self.val_errors.append(val_outs[0])

                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
                    
            callbacks.on_epoch_end(epoch, epoch_logs)                    
      
            this_train_loss = np.average(train_losses)      
            self.errors.append(this_train_loss)
           
                  
            ### autimatically saving snapshot ..
            if np.mod(epoch+1,self.snapshot)==0:
                if epoch is not self.n_epochs: self.save()


        callbacks.on_train_end()   


    def train_on_batch(self,X_train,X_mask,Y_train,verbose=1,shuffle=True, show_accuracy=True):
        train_set_x = np.asarray(X_train)
        train_set_y = np.asarray(Y_train)
	
        if X_mask is None:
            ins = [train_set_x, train_set_y]
        else:
            mask_set_x = np.asarray(X_mask)            
            ins = [train_set_x, mask_set_x, train_set_y]
            

        if show_accuracy:
            return self._train_with_acc(*ins)
        else:
            return self._train(*ins)

    def test_on_batch(self,X_val,X_val_mask,Y_val,verbose=1,shuffle=True, show_accuracy=True):

        val_set_x = np.asarray(X_val)
        val_set_y = np.asarray(Y_val)   
          
        
        if X_val_mask is None:
            val_ins = [val_set_x, val_set_y]
        else:
            mask_val_set_x = np.asarray(X_val_mask)               
            val_ins = [val_set_x, mask_val_set_x, val_set_y]
            
        if show_accuracy:
            return self._test_with_acc(*val_ins)
        else:
            return self._test(*val_ins)



    def evaluate(self, X_test,X_mask,Y_test, batch_size=128, show_accuracy=False, verbose=1):
        
        test_set_x = np.asarray(X_test)
        test_set_y = np.asarray(Y_test)

        
        if mask_set_x is None:
            ins = [test_set_x, test_set_y]
        else:
            mask_set_x = np.asarray(X_mask, dtype='float32')
            ins = [test_set_x, mask_set_x, test_set_y]
            
        if show_accuracy:
            f = self._test_with_acc
        else:
            f = self._test
        outs = self._test_loop(f, ins, batch_size, verbose)
        if show_accuracy:
            return outs
        else:
            return outs[0]

    def predict(self, X_test,X_mask):

        X_test = np.asarray(X_test, dtype='int32')
        X_mask = np.asarray(X_mask, dtype='float32')    

        if X_test.ndim==1: X_test=X_test[None,:]
        if X_mask.ndim==1: X_mask=X_mask[None,:]        
        
        ins=[X_test,X_mask]
    
        result = np.argmax(self._predict(*ins), axis = -1)[0]
        
        return result

    def _predict_loop(self, f, ins, batch_size=128, verbose=0):
        '''
            Abstract method to loop over some data in batches.
        '''
        nb_sample = ins[0].shape[0]
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)
            print('loading images...')
            ins_batch[0]=load_image(ins[0])
            
            batch_outs = f(*ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        return outs

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''
            Abstract method to loop over some data in batches.
        '''
        nb_sample = ins[0].shape[0]
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)
            
            if len(ins_batch)==3:
                ins_batch=[load_image(ins_batch[0]),ins_batch[1],ins_batch[2]]
            else: ins_batch=[load_image(ins_batch[0]),ins_batch[1]]

            batch_outs = f(*ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        return outs
