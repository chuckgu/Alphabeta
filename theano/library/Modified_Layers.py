import theano
import theano.tensor as T
import numpy as np
from Initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX,orthogonal,one,uniform
import theano.typed_list
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Activations import relu,LeakyReLU,tanh,sigmoid,linear,mean,max,softmax,hard_sigmoid
from Recurrent_Layers import Recurrent
from Layers import dropout_layer


class SGRU(Recurrent):
    def __init__(self,n_in,n_hidden,n_seg=4,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_seg=int(n_seg)
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.return_seq=return_seq
        
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        
        self.W_z1 = glorot_uniform((n_in,n_hidden/4))        
        self.b_z1 = zero((n_hidden/4,))
        self.W_z2 = glorot_uniform((n_in,n_hidden/4))        
        self.b_z2 = zero((n_hidden/4,))
        self.W_z3 = glorot_uniform((n_in,n_hidden/4))        
        self.b_z3 = zero((n_hidden/4,))
        self.W_z4 = glorot_uniform((n_in,n_hidden/4))        
        self.b_z4 = zero((n_hidden/4,))        
    

        self.U_r = glorot_uniform((n_hidden,n_hidden))
        
        self.W_r1 = glorot_uniform((n_in,n_hidden/4))        
        self.b_r1 = zero((n_hidden/4,))
        self.W_r2 = glorot_uniform((n_in,n_hidden/4))        
        self.b_r2 = zero((n_hidden/4,))
        self.W_r3 = glorot_uniform((n_in,n_hidden/4))        
        self.b_r3 = zero((n_hidden/4,))
        self.W_r4 = glorot_uniform((n_in,n_hidden/4))        
        self.b_r4 = zero((n_hidden/4,))
        

        self.U_h = glorot_uniform((n_hidden,n_hidden))
        
        self.W_h1 = glorot_uniform((n_in,n_hidden/4)) 
        self.b_h1 = zero((n_hidden/4,))
        self.W_h2 = glorot_uniform((n_in,n_hidden/4)) 
        self.b_h2 = zero((n_hidden/4,))
        self.W_h3 = glorot_uniform((n_in,n_hidden/4)) 
        self.b_h3 = zero((n_hidden/4,))        
        self.W_h4 = glorot_uniform((n_in,n_hidden/4)) 
        self.b_h4 = zero((n_hidden/4,))        
        
        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        self.L1 = 0
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)         

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z))
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        return h_t

    def get_output(self, train=False, init_state=None):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))
        
        x_z = T.concatenate([T.dot(X, self.W_z1) + self.b_z1, T.dot(X, self.W_z2) + self.b_z2,T.dot(X, self.W_z3) + self.b_z3,T.dot(X, self.W_z4) + self.b_z4], axis=-1)
        x_r = T.concatenate([T.dot(X, self.W_r1) + self.b_r1, T.dot(X, self.W_r2) + self.b_r2,T.dot(X, self.W_r3) + self.b_r3,T.dot(X, self.W_r4) + self.b_r4], axis=-1)
        x_z = T.concatenate([T.dot(X, self.W_h1) + self.b_h1, T.dot(X, self.W_h2) + self.b_h2,T.dot(X, self.W_h3) + self.b_h3,T.dot(X, self.W_h4) + self.b_h4], axis=-1)


        if init_state is None: init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)        
        
        h, c = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=init_state,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_seq is False: 
            h[-1]
        return h.dimshuffle((1, 0, 2))
        

class Attention2(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',mode='soft'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)

        self.input= T.tensor3()
        self.input2=T.matrix()        
        self.x_mask=T.matrix()
        #self.activation=eval(activation)
        self.mode=mode

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.b_h = zero((n_hidden,))

        self.W_c = glorot_uniform((4096,n_hidden)) 
        self.b_c = zero((n_hidden,))

        self.W_v = glorot_uniform((n_hidden,n_hidden)) 
        self.W_l = glorot_uniform((n_hidden,n_hidden))    
        
        self.W_lh = glorot_uniform((n_hidden,n_hidden)) 
        self.W_vh = glorot_uniform((n_hidden,n_hidden)) 

        self.U_att= orthogonal((n_hidden,1)) 
        self.b_att= zero((1,))

        self.params=[self.W_h,self.b_h,self.W_c,self.b_c,self.W_v,self.W_l,self.U_att,self.b_att,self.W_lh,self.W_vh]
        
        self.L1 = 0
        self.L2_sqr = 0

    def add_input(self, add_input=None):
        self.input2=add_input

    
    def _step(self,h_tm1,p_x,p_xm,ctx):
        #visual attention
    
        #ctx=dropout_layer(ctx)
        v_a=T.exp(ctx+T.dot(h_tm1,self.W_v))
        v_a=v_a/v_a.sum(1, keepdims=True) 
        
        ctx_p=ctx*v_a
    
        #linguistic attention
        l_a=p_x+T.dot(h_tm1,self.W_l)[None,:,:]

        l_a=T.dot(l_a,self.U_att)+self.b_att        

        l_a=T.exp(l_a.reshape((l_a.shape[0],l_a.shape[1])))
        
        l_a=l_a/l_a.sum(0, keepdims=True) 
        
        l_a=l_a*p_xm
        
        p_x_p=(p_x*l_a[:,:,None]).sum(0)
        
        h= T.dot(ctx_p,self.W_vh) + T.dot(p_x_p,self.W_lh)

        return h

    
    def get_output(self,train=False):
        if self.mode is 'soft':

            X = self.get_input(train)
            padded_mask = self.get_mask().astype('int8')
            X = X.dimshuffle((1, 0, 2))
            padded_mask = padded_mask.dimshuffle((1, 0))

            p_x = T.dot(X, self.W_h) + self.b_h
            ctx = T.dot(self.input2, self.W_c) + self.b_c           
            
            ctx=dropout_layer(ctx,0.25)
            
            h, _ = theano.scan(self._step, 
                                 #sequences = [X],
                                 outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
                                 non_sequences=[p_x,padded_mask,ctx],
                                 n_steps=X.shape[0] )
        return h[-1]                                                                    

class Attention3(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',mode='soft'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)

        self.input= T.tensor3()
        self.input2=T.matrix()        
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.mode=mode

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.b_h = zero((n_hidden,))

        self.W_c = glorot_uniform((4096,n_hidden)) 
        self.b_c = zero((n_hidden,))

        self.W_v = glorot_uniform((n_hidden,n_hidden))     
        
        self.params=[self.W_h,self.b_h,self.W_c,self.b_c,self.W_v]
        
        self.L1 = 0
        self.L2_sqr = 0

    def add_input(self, add_input=None):
        self.input2=add_input


    def get_output(self,train=False):
        if self.mode is 'soft':

            X=self.get_input(train)
            
            img=T.dot(self.input2,self.W_c)+self.b_c
            
            output=self.activation(T.dot(X,self.W_h)+self.b_h+img)
            
            output=T.dot(output,self.W_v)

            #x_mask=self.x_mask.astype('int8')
            
            e=T.exp(output)
            
            e=e/e.sum(1, keepdims=True)
            
            #e=e*x_mask
      
            output=(img*e)+X
                
        return  output                                                           


class GRU2(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.input2=T.matrix()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.return_seq=return_seq
        
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))

        self.W_c = glorot_uniform((4096,n_hidden)) 
        self.b_c = zero((n_hidden,))
        self.W_hc=glorot_uniform((n_hidden,n_hidden))
        
        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.W_c, self.b_c#, self.W_hc
        ]

        self.L1 = 0
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)         

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h, ctx):
                  
         
        ctx=dropout_layer(ctx)
        
        c=ctx#+T.dot(h_tm1,self.W_hc) 
        
            
    
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z)+c)
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r)+c)
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h)+c)
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        
        
        return h_t
        
    def add_input(self, add_input=None):
        self.input2=add_input

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))
             
        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        ctx = T.dot(self.input2, self.W_c) + self.b_c
       
        init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
        #init_state=ctx
        h, _ = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=init_state,
            non_sequences=[self.U_z, self.U_r, self.U_h, ctx])

        if self.return_seq is False: return h[-1]
        return h.dimshuffle((1, 0, 2))

class GRU3(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.input2=T.matrix()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.return_seq=return_seq
        
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))

        self.W_c = glorot_uniform((4096,n_hidden)) 
        self.b_c = zero((n_hidden,))
        self.W_hc=glorot_uniform((n_hidden,n_hidden))
        self.W_hl=glorot_uniform((n_hidden,n_hidden))
        self.W_cl=glorot_uniform((n_hidden,n_hidden))
        
        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.W_c, self.b_c, self.W_hc,
            self.W_hl,self.W_cl
        ]

        self.L1 = 0
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)         

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,l_tm1,
              u_z, u_r, u_h, ctx):
                  
        c=ctx+T.dot(h_tm1,self.W_hc) 
        
        c=tanh(c)

        c=T.exp(c)
        
        c=c/c.sum(-1, keepdims=True)
  
        c=ctx*c               
    
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z)+c)
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r)+c)
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h)+c)
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1+c
        
        logit=tanh(T.dot(h_t, self.W_hl)+T.dot(c, self.W_cl))        
        
        return h_t,logit
        
    def add_input(self, add_input=None):
        self.input2=add_input

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))
        
        ctx=dropout_layer(self.input2,0.25) 
        
        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        ctx = T.dot(ctx, self.W_c) + self.b_c


        
        init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
        [h,logit], _ = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=[init_state,init_state],
            non_sequences=[self.U_z, self.U_r, self.U_h,ctx])

        if self.return_seq is False: return logit[-1]
        return logit.dimshuffle((1, 0, 2))




class LSTM2(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.input2=T.matrix()
        self.x_mask=T.matrix()
        self.activation=eval(activation)    
        self.return_seq=return_seq
        
        self.W_i = glorot_uniform((n_in,n_hidden))
        self.U_i = orthogonal((n_hidden,n_hidden))
        self.b_i = zero((n_hidden,))

        self.W_f = glorot_uniform((n_in,n_hidden))
        self.U_f = orthogonal((n_hidden,n_hidden))
        self.b_f = one((n_hidden,))

        self.W_c = glorot_uniform((n_in,n_hidden))
        self.U_c = orthogonal((n_hidden,n_hidden))
        self.b_c = zero((n_hidden,))

        self.W_o = glorot_uniform((n_in,n_hidden))
        self.U_o = orthogonal((n_hidden,n_hidden))
        self.b_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]
        
        self.L1 = 0
        self.L2_sqr = 0
        
        
    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):

        i_t = hard_sigmoid(xi_t + T.dot(h_tm1, u_i))
        f_t = hard_sigmoid(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        c_t = mask_tm1 * c_t + (1. - mask_tm1) * c_tm1        
        
        o_t = hard_sigmoid(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        h_t = mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        
        return h_t, c_t


    def add_input(self, add_input=None):
        self.input2=add_input
    
    def get_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))


        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o
        
        init_state=self.input2

        [h, c], _ = theano.scan(self._step,
                                sequences=[xi, xf, xo, xc, padded_mask],
                                outputs_info=[
                                    init_state,
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
                                ],
                                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c])        

        if self.return_seq is False: return h[-1]                                                                 
        return h.dimshuffle((1, 0, 2))
        
        
class BiDirectionGRU2(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',output_mode='concat',return_seq=True):
        self.n_in=int(n_in)
        if output_mode is 'concat':n_hidden=int(n_hidden/2)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
        self.input2=T.matrix()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.return_seq=return_seq
        
        # forward weights
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))

        self.W_c = glorot_uniform((4096,n_hidden)) 
        self.b_c = zero((n_hidden,))

        
        # backward weights
        self.Wb_z = glorot_uniform((n_in,n_hidden))
        self.Ub_z = glorot_uniform((n_hidden,n_hidden))
        self.bb_z = zero((n_hidden,))

        self.Wb_r = glorot_uniform((n_in,n_hidden))
        self.Ub_r = glorot_uniform((n_hidden,n_hidden))
        self.bb_r = zero((n_hidden,))

        self.Wb_h = glorot_uniform((n_in,n_hidden)) 
        self.Ub_h = glorot_uniform((n_hidden,n_hidden))
        self.bb_h = zero((n_hidden,))        

        self.Wb_c = glorot_uniform((4096,n_hidden)) 
        self.bb_c = zero((n_hidden,))


        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.W_c, self.b_c,

            self.Wb_z, self.Ub_z, self.bb_z,
            self.Wb_r, self.Ub_r, self.bb_r,
            self.Wb_h, self.Ub_h, self.bb_h,
            self.Wb_c, self.bb_c
        ]

        self.L1 = T.sum(abs(self.W_z))+T.sum(abs(self.U_z))+\
                  T.sum(abs(self.W_r))+T.sum(abs(self.U_r))+\
                  T.sum(abs(self.W_h))+T.sum(abs(self.U_h))+\
                  T.sum(abs(self.Wb_z))+T.sum(abs(self.Ub_z))+\
                  T.sum(abs(self.Wb_r))+T.sum(abs(self.Ub_r))+\
                  T.sum(abs(self.Wb_h))+T.sum(abs(self.Ub_h))
        
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)+\
                      T.sum(self.Wb_z**2) + T.sum(self.Ub_z**2)+\
                      T.sum(self.Wb_r**2) + T.sum(self.Ub_r**2)+\
                      T.sum(self.Wb_h**2) + T.sum(self.Ub_h**2)

    def _fstep(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h,
              ctx):

        ctx=dropout_layer(ctx)                
            
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z)+ctx)
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r)+ctx)
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h)+ctx)
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        return h_t



    def _bstep(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h,
              ctx):

        ctx=dropout_layer(ctx)                
            
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z)+ctx)
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r)+ctx)
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h)+ctx)
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        return h_t
 
       

    def get_forward_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        ctx = T.dot(self.input2, self.W_c) + self.b_c        
        #init_state=self.input2    
        init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
        
        h, c = theano.scan(
            self._fstep,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=init_state,
            non_sequences=[self.U_z, self.U_r, self.U_h,ctx])

        if self.return_seq is False: return h[-1]
        return h.dimshuffle((1, 0, 2))        
        
        
    def get_backward_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.Wb_z) + self.bb_z
        x_r = T.dot(X, self.Wb_r) + self.bb_r
        x_h = T.dot(X, self.Wb_h) + self.bb_h
        ctx = T.dot(self.input2, self.Wb_c) + self.bb_c        
        #init_state=self.input2    
        init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)     

        h, c = theano.scan(
            self._bstep,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=init_state,
            non_sequences=[self.Ub_z, self.Ub_r, self.Ub_h, ctx],go_backwards = True)

        if self.return_seq is False: return h[-1]
        return h.dimshuffle((1, 0, 2))                

    def add_input(self, add_input=None):
        self.input2=add_input

    def get_output(self,train=False):
        forward = self.get_forward_output(train)
        backward = self.get_backward_output(train)
        if self.output_mode is 'sum':
            return forward + backward
        elif self.output_mode is 'concat':
            if self.return_seq: axis=2
            else: axis=1
            return T.concatenate([forward, backward], axis=axis)
        else:
            raise Exception('output mode is not sum or concat')