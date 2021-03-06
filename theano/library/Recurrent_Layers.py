import theano
import theano.tensor as T
import numpy as np
from Initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX,orthogonal,one,uniform
import theano.typed_list
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Activations import relu,LeakyReLU,tanh,sigmoid,linear,mean,max,softmax,hard_sigmoid

class Recurrent(object):
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

class Hidden(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)        
        
        self.W_hh=glorot_uniform((n_hidden,n_hidden))
        self.W_in=glorot_uniform((n_in,n_hidden))
        self.bh=zero((n_hidden,))
        
        self.params=[self.W_hh,self.W_in,self.bh]
        
        
        self.L1 = T.sum(abs(self.W_hh))+T.sum(abs(self.W_in))
        self.L2_sqr = T.sum(self.W_hh**2) + T.sum(self.W_in**2)

    
    def _step(self,x_t,x_m, h_tm1):
        h=self.activation(T.dot(h_tm1, self.W_hh) + T.dot(x_t, self.W_in) + self.bh)
        h=h*x_m[:,None]
        return h

    def get_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask
        X = X.dimshuffle((1, 0, 2))
        X_mask = X_mask.dimshuffle((1, 0))
        h, _ = theano.scan(self._step, 
                             sequences = [X,X_mask],
                             outputs_info = alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden))
        if self.return_seq is False: return h[-1]                                                                    
        return h.dimshuffle((1, 0, 2))
        


class LSTM(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
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

    
    def get_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))


        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [h, c], _ = theano.scan(self._step,
                                sequences=[xi, xf, xo, xc, padded_mask],
                                outputs_info=[
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
                                ],
                                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c])        

        if self.return_seq is False: return h[-1]                                                                 
        return h.dimshuffle((1, 0, 2))




class GRU(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',return_seq=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
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
        
        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        
        if init_state is None: init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)        
        
        h, c = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=init_state,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_seq is False: return h[-1]
        return h.dimshuffle((1, 0, 2))
        

class BiDirectionLSTM(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',output_mode='concat',return_seq=True):
        self.n_in=int(n_in)
        if output_mode is 'concat': n_hidden=int(n_hidden/2)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        self.return_seq=return_seq
        
        # forward weights
        self.W_i = glorot_uniform((n_in,n_hidden))
        self.U_i = glorot_uniform((n_hidden,n_hidden))
        self.b_i = zero((n_hidden,))

        self.W_f = glorot_uniform((n_in,n_hidden))
        self.U_f = glorot_uniform((n_hidden,n_hidden))
        self.b_f = zero((n_hidden,))

        self.W_c = glorot_uniform((n_in,n_hidden))
        self.U_c = glorot_uniform((n_hidden,n_hidden))
        self.b_c = zero((n_hidden,))

        self.W_o = glorot_uniform((n_in,n_hidden))
        self.U_o = glorot_uniform((n_hidden,n_hidden))
        self.b_o = zero((n_hidden,))
        
        # backward weights
        self.Wb_i = glorot_uniform((n_in,n_hidden))
        self.Ub_i = glorot_uniform((n_hidden,n_hidden))
        self.bb_i = zero((n_hidden,))

        self.Wb_f = glorot_uniform((n_in,n_hidden))
        self.Ub_f = glorot_uniform((n_hidden,n_hidden))
        self.bb_f = zero((n_hidden,))

        self.Wb_c = glorot_uniform((n_in,n_hidden))
        self.Ub_c = glorot_uniform((n_hidden,n_hidden))
        self.bb_c = zero((n_hidden,))
        
        self.Wb_o = glorot_uniform((n_in,n_hidden))
        self.Ub_o = glorot_uniform((n_hidden,n_hidden))
        self.bb_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,

            self.Wb_i, self.Ub_i, self.bb_i,
            self.Wb_c, self.Ub_c, self.bb_c,
            self.Wb_f, self.Ub_f, self.bb_f,
            self.Wb_o, self.Ub_o, self.bb_o,
        ]

        self.L1 = 0
        self.L2_sqr = 0    
    
    
        
    def _fstep(self,
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


    def _bstep(self,
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
          
    
    def get_forward_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))
        
        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [h, c], _ = theano.scan(self._fstep,
                                sequences=[xi, xf, xo, xc, padded_mask],
                                outputs_info=[
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
                                ],
                                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c])        

        if self.return_seq is False: return h[-1]                                                                 
        return h.dimshuffle((1, 0, 2))        

    def get_backward_output(self,train=False):
        X = self.get_input(train)
        padded_mask = self.get_mask()[:,:, None].astype('int8')
        X = X.dimshuffle((1, 0, 2))
        padded_mask = padded_mask.dimshuffle((1, 0, 2))
        
        xi = T.dot(X, self.Wb_i) + self.bb_i
        xf = T.dot(X, self.Wb_f) + self.bb_f
        xc = T.dot(X, self.Wb_c) + self.bb_c
        xo = T.dot(X, self.Wb_o) + self.bb_o

        [h, c], _ = theano.scan(self._bstep,
                                sequences=[xi, xf, xo, xc, padded_mask],
                                outputs_info=[
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
                                    T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
                                ],
                                non_sequences=[self.Ub_i, self.Ub_f, self.Ub_o, self.Ub_c],
                                go_backwards = True)        

        if self.return_seq is False: return h[-1]                                                                 
        return h.dimshuffle((1, 0, 2))               



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


class BiDirectionGRU(Recurrent):
    def __init__(self,n_in,n_hidden,activation='tanh',output_mode='concat',return_seq=True):
        self.n_in=int(n_in)
        if output_mode is 'concat':n_hidden=int(n_hidden/2)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
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

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,

            self.Wb_z, self.Ub_z, self.bb_z,
            self.Wb_r, self.Ub_r, self.bb_r,
            self.Wb_h, self.Ub_h, self.bb_h
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
              u_z, u_r, u_h):
        
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z))
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=mask_tm1 * h_t + (1. - mask_tm1) * h_tm1
        return h_t



    def _bstep(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        
        z = hard_sigmoid(xz_t + T.dot(h_tm1, u_z))
        r = hard_sigmoid(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
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
        h, c = theano.scan(
            self._fstep,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h])

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
        h, c = theano.scan(
            self._bstep,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1),
            non_sequences=[self.Ub_z, self.Ub_r, self.Ub_h],go_backwards = True)

        if self.return_seq is False: return h[-1]
        return h.dimshuffle((1, 0, 2))                

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

class Decoder(Recurrent):
    def __init__(self,n_in,n_hidden,n_out,activation='tanh'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.input= T.tensor3()
        self.output= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)

        
        self.W_z = glorot_uniform((n_out,n_hidden))
        self.U_z = orthogonal((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_out,n_hidden))
        self.U_r = orthogonal((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_out,n_hidden)) 
        self.U_h = orthogonal((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))
        
        #self.U_att= orthogonal((self.n_in,1)) 
        #self.b_att= zero((1,))

        self.W_yc=glorot_uniform((self.n_out,))
        

        self.W_cy = glorot_uniform((self.n_in,self.n_hidden))
        self.W_cs= glorot_uniform((self.n_in,self.n_hidden))

        
        self.W_ha = glorot_uniform((self.n_in,self.n_in))
        #self.W_sa= glorot_uniform((self.n_hidden,self.n_in))
        

        
        self.W_cl= glorot_uniform((self.n_in,self.n_out))
        self.W_yl= glorot_uniform((self.n_out,self.n_out))
        self.W_hl= glorot_uniform((self.n_hidden,self.n_out))
        
        self.params=[self.W_z,self.U_z,self.b_z,self.W_r,self.U_r,self.b_r,
                   self.W_h,self.U_h,self.b_h,self.W_cy,self.W_cs,self.W_ha
                     ,self.W_cl,self.W_yl,self.W_hl]
        

        self.L1 = 0
        self.L2_sqr = 0
        
    def _step(self,y_tm1,yz_t, yr_t, yh_t,y_m,s_tm1,h,x_m):
        
        # no attention
        c= (h * x_m[:,:,None]).sum(0) / x_m.sum(0)[:,None]
        c=T.dot(c,self.W_ha)

        z = hard_sigmoid(yz_t + T.dot(s_tm1, self.U_z)+T.dot(c,self.W_cs))
        r = hard_sigmoid(yr_t + T.dot(s_tm1, self.U_r)+T.dot(c,self.W_cs))
        hh_t = self.activation(yh_t + T.dot(r * s_tm1, self.U_h)+T.dot(c,self.W_cy))
        s_t = z * s_tm1 + (1 - z) * hh_t
        
        s_t = (1. - y_m)[:,None] * s_tm1 + y_m[:,None] * s_t
        
        logit=self.activation(T.dot(s_t, self.W_hl)+T.dot(y_tm1, self.W_yl)+T.dot(c, self.W_cl))
        
        return T.cast(s_t,dtype =theano.config.floatX),T.cast(logit,dtype =theano.config.floatX)  



    def get_sample(self,y,s_tm1):
        c=self.get_input()
        X_mask=self.previous.x_mask
        y_mask=T.alloc(1.,y.shape[0])
        
        c = c.dimshuffle((1, 0, 2))
        X_mask = X_mask.dimshuffle((1, 0))

        y_z = T.dot(y, self.W_z) + self.b_z
        y_r = T.dot(y, self.W_r) + self.b_r
        y_h = T.dot(y, self.W_h) + self.b_h  

        
        h,logit=self._step(y,y_z,y_r,y_h,y_mask,s_tm1,c,X_mask) 
        

        return h,logit
    
    def get_output(self,y,y_mask,init_state,train=False):
        X=self.get_input(train)  
        X_mask=self.previous.x_mask
        X = X.dimshuffle((1, 0, 2))
        X_mask = X_mask.dimshuffle((1, 0))  
        
        y=y.dimshuffle((1, 0, 2))
        y_mask=y_mask.dimshuffle((1, 0))  
        
        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y)
        y_shifted=T.set_subtensor(y_shifted[1:],y[:-1])
        y=y_shifted 

        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y_mask)
        y_shifted=T.set_subtensor(y_shifted[1:],y_mask[:-1])
        y_mask=y_shifted 

        y_z = T.dot(y, self.W_z) + self.b_z
        y_r = T.dot(y, self.W_r) + self.b_r
        y_h = T.dot(y, self.W_h) + self.b_h       
        
        #init_state=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_hidden), 1)
        
        [h,logit], _ = theano.scan(self._step, 
                                     sequences = [y,y_z,y_r,y_h,y_mask],
                                     outputs_info = [init_state,
                                                     None],
                                     non_sequences=[X,X_mask])

        return logit.dimshuffle((1, 0, 2))
        


class Decoder_attention(Recurrent):
    def __init__(self,n_in,n_hidden,n_out,activation='tanh'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.input= T.tensor3()
        self.output= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)

        
        self.W_z = glorot_uniform((n_out,n_hidden))
        self.U_z = orthogonal((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_out,n_hidden))
        self.U_r = orthogonal((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_out,n_hidden)) 
        self.U_h = orthogonal((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))
        
        self.U_att= orthogonal((self.n_in,1)) 
        self.b_att= zero((1,))

        self.W_yc=glorot_uniform((self.n_out,))
        

        self.W_cy = glorot_uniform((self.n_in,self.n_hidden))
        self.W_cs= glorot_uniform((self.n_in,self.n_hidden))

        
        self.W_ha = glorot_uniform((self.n_in,self.n_in))
        self.W_sa= glorot_uniform((self.n_hidden,self.n_in))
        

        
        self.W_cl= glorot_uniform((self.n_in,self.n_out))
        self.W_yl= glorot_uniform((self.n_out,self.n_out))
        self.W_hl= glorot_uniform((self.n_hidden,self.n_out))
        
        self.params=[self.W_z,self.U_z,self.b_z,self.W_r,self.U_r,self.b_r,
                   self.W_h,self.U_h,self.b_h,self.W_cy,self.W_cs,self.W_ha,self.W_sa
                     ,self.W_cl,self.W_yl,self.W_hl,self.U_att,self.b_att]
        

        self.L1 = 0
        self.L2_sqr = 0
        
    def _step(self,y_tm1,yz_t, yr_t, yh_t,y_m,s_tm1,h,x_m):
        
                    
        # attention
        
        pctx__=T.dot(h,self.W_ha)+T.dot(s_tm1,self.W_sa)[None,:,:]
        
        pctx__=self.activation(pctx__)
        
        e=T.dot(pctx__,self.U_att)+self.b_att
        
        e=T.exp(e.reshape((e.shape[0],e.shape[1])))
        
        e=e/e.sum(0, keepdims=True)
        
        e=e*x_m
  
        c=(h*e[:,:,None]).sum(0)


        z = hard_sigmoid(yz_t + T.dot(s_tm1, self.U_z)+T.dot(c,self.W_cs))
        r = hard_sigmoid(yr_t + T.dot(s_tm1, self.U_r)+T.dot(c,self.W_cs))
        hh_t = self.activation(yh_t + T.dot(r * s_tm1, self.U_h)+T.dot(c,self.W_cy))
        s_t = z * s_tm1 + (1 - z) * hh_t
        
        s_t = (1. - y_m)[:,None] * s_tm1 + y_m[:,None] * s_t
        
        logit=self.activation(T.dot(s_t, self.W_hl)+T.dot(y_tm1, self.W_yl)+T.dot(c, self.W_cl))
        
        return T.cast(s_t,dtype =theano.config.floatX),T.cast(logit,dtype =theano.config.floatX)  



    def get_sample(self,y,s_tm1):
        c=self.get_input()
        X_mask=self.previous.x_mask
        y_mask=T.alloc(1.,y.shape[0])
        
        c = c.dimshuffle((1, 0, 2))
        X_mask = X_mask.dimshuffle((1, 0))

        y_z = T.dot(y, self.W_z) + self.b_z
        y_r = T.dot(y, self.W_r) + self.b_r
        y_h = T.dot(y, self.W_h) + self.b_h  

        
        h,logit=self._step(y,y_z,y_r,y_h,y_mask,s_tm1,c,X_mask) 
        

        return h,logit
    
    def get_output(self,y,y_mask,init_state,train=False):
        X=self.get_input(train)  
        X_mask=self.previous.x_mask
        X = X.dimshuffle((1, 0, 2))
        X_mask = X_mask.dimshuffle((1, 0))  
        
        y=y.dimshuffle((1, 0, 2))
        y_mask=y_mask.dimshuffle((1, 0))  
        
        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y)
        y_shifted=T.set_subtensor(y_shifted[1:],y[:-1])
        y=y_shifted 

        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y_mask)
        y_shifted=T.set_subtensor(y_shifted[1:],y_mask[:-1])
        y_mask=y_shifted 

        y_z = T.dot(y, self.W_z) + self.b_z
        y_r = T.dot(y, self.W_r) + self.b_r
        y_h = T.dot(y, self.W_h) + self.b_h       
        
        
        
        [h,logit], _ = theano.scan(self._step, 
                                     sequences = [y,y_z,y_r,y_h,y_mask],
                                     outputs_info = [init_state,
                                                     None],
                                     non_sequences=[X,X_mask])

        return logit.dimshuffle((1, 0, 2))