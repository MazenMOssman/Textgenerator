#!/usr/bin/env python
# coding: utf-8

# In[63]:

import tensorflow as tf
class mask():
    
    def __init__(self, inputs):
        self.batch_size, self.input_length =  batch_size, input_length = tf.shape(inputs)[0] , tf.shape(inputs)[1]
        
    def __call__(self):
        
        mask = tf.cast(tf.range(self.input_length)[:,tf.newaxis]> tf.range(self.input_length), dtype = 'int32')
        mask = tf.reshape(mask, (1,self.input_length, self.input_length))
        mul = tf.concat([tf.expand_dims(self.batch_size,-1), tf.constant([1,1])], axis = 0)
        return tf.tile(mask, mul)

