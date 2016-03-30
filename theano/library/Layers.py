import theano
import theano.tensor as T
import numpy as np
from Initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX,orthogonal,one,uniform
import theano.typed_list
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Activations import relu,LeakyReLU,tanh,sigmoid,linear,mean,max,softmax,hard_sigmoid


def dropout_layer(X, train=True, trng=RandomStreams(seed=np.random.randint(10e6)),pr=0.5):
    if pr > 0.:
        retain_prob = 1. - pr
        if train:
            X *= trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        else:
            X *= retain_prob
    return X                

def make_tuple(*args):
    return args

class Layer(object):
    
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
    def set_input(self,x):
        self.input=x

    def get_mask(self):
        return self.x_mask
            
    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    

            
            
class Embedding(Layer):
    def __init__(self,n_in,n_hidden,n_out=0,n_out_hidden=0,multi=False,shared_emb=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.imatrix()
        self.x_mask=T.imatrix()
        self.multi=multi
        self.W=uniform((n_in,n_hidden))
        self.shared_emb=shared_emb
        if multi:
            self.y= T.imatrix()
            self.y_mask=T.imatrix()
                

        if shared_emb:
            self.params=[self.W]
        else:
            self.n_out=int(n_out)
            self.n_out_hidden=int(n_out_hidden)            
            self.W_multi=uniform((n_out,n_out_hidden))
            self.params=[self.W,self.W_multi]
        
        self.L1 = 0
        self.L2_sqr = 0

    def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out

    def set_input_y(self,y):
        self.y=y

    def get_input_y(self):
        return self.y

    def set_mask_y(self,y_mask):
        self.y_mask=y_mask

    def get_mask_y(self):
        return self.y_mask

    def get_multi_output(self):
        y = self.y
        if self.shared_emb: out = self.W[y]
        else: out = self.W_multi[y]
        return out


class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        self.activation = eval(activation)
        self.target = target
        self.beta = beta
        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0
 

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

class Drop_out(Layer):
    def __init__(self,pr=0.5):
        self.input= T.tensor3()
        self.x_mask=T.matrix()   
        self.trng = RandomStreams(seed=np.random.randint(10e6))
        self.params=[]
        self.L1=0
        self.L2_sqr=0
        self.pr=pr
        
    
    def get_output(self,train=False):
        X = self.get_input(train)
        if self.pr > 0.:
            retain_prob = 1. - self.pr
            if train:
                X *= self.trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X        


class Pool(Layer):
    def __init__(self,mode='mean'):
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        #self.activation=eval(activation)
        self.mode=mode

        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0

    
    def get_output(self,train=False):
        if self.mode is 'mean':
            X=self.get_input(train)
            proj = (X * self.x_mask[:, :, None]).sum(axis=1)
            output = proj / self.x_mask.sum(axis=1)[:, None]    
        elif self.mode is 'final':
            X=self.get_input(train)
            proj = (X * self.x_mask[:, :, None])
            output=proj[self.x_mask.sum(axis=0),T.arange(proj.shape[1])]
        return output


class FC_layer(Layer):
    def __init__(self,n_in,n_hidden,activation='linear'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        
        
        self.W=glorot_uniform((n_in,n_hidden))
        self.b=zero((n_hidden,))

        
        self.params=[self.W,self.b]
        
        self.L1 = T.sum(abs(self.W))+T.sum(abs(self.b))
        self.L2_sqr = T.sum(self.W**2)+T.sum(self.b**2)
  
    
    def get_output(self,train=False):
        X=self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output     

class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self,dim=2):
        super(Flatten, self).__init__()
        self.params=[]
        self.input= T.tensor3()
        self.x_mask=T.matrix() 
        self.dim=dim
        self.L1=0
        self.L2_sqr=0

    def get_output(self, train=False):
        X = self.get_input(train)
	
        if self.dim == 3:
            size = theano.tensor.prod(X.shape) // (X.shape[0]*X.shape[1])
            nshape = (X.shape[0],X.shape[1], size)
        else:
            size = theano.tensor.prod(X.shape) // X.shape[0]
            nshape = (X.shape[0], size)            
        return theano.tensor.reshape(X, nshape)

class Flatten_3d(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self,dim=2):
        super(Flatten_3d, self).__init__()
        self.params=[]
        dtensor5 = T.TensorType('float32', (False,)*5)
        self.input = dtensor5()
        self.x_mask=T.matrix() 
        self.dim=dim
        self.L1=0
        self.L2_sqr=0

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim == 3:
            size = theano.tensor.prod(X.shape) // (X.shape[0]*X.shape[1])
            nshape = (X.shape[0],X.shape[1], size)
        else:
            size = theano.tensor.prod(X.shape) // X.shape[0]
            nshape = (X.shape[0], size)            
        return theano.tensor.reshape(X, nshape)


class Reshape(Layer):
    '''
        Reshape an output to a certain shape.
        Can't be used as first layer in a model (no fixed input!)
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims

        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0


    def get_output(self, train=False):
        X = self.get_input(train)
        nshape = make_tuple(X.shape[0], *self.dims)
        return theano.tensor.reshape(X, nshape)


class RepeatVector(Layer):
    '''
        Repeat input n times.

        Dimensions of input are assumed to be (nb_samples, dim).
        Return tensor of shape (nb_samples, n, dim).
    '''
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0
        
    def get_output(self, train=False):
        X = self.get_input(train)
        tensors = [X]*self.n
        stacked = theano.tensor.stack(*tensors)
        return stacked.dimshuffle((1, 0, 2))
      
        
class Attention(Layer):
    def __init__(self,mode='soft'):
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        #self.activation=eval(activation)
        self.mode=mode

        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0

    
    def get_output(self,train=False):
        if self.mode is 'soft':
            X=self.get_input(train)
            
            x_mask=self.x_mask[:,:, None].astype('int8')
            
            e=T.exp(X)
            
            e=e/e.sum(1, keepdims=True)
            
            e=e*x_mask
      
            output=(X*e).sum(1)           
                
        return output        
