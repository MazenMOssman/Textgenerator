#!/usr/bin/env python
# coding: utf-8

# In[118]:


import tensorflow as tf
class attention_block(tf.keras.layers.Layer):
    """
    A class that returns attention blocks.
    
    Parameters: att_head: the attention head
                head_size: the size head
                dense_size: activation dense_size
                output_size: the output size
                
    Returns: Attention blocks of transformer
    
    """
    def __init__(self, att_head, head_size, dense_size, output_size):
        
        super().__init__()
        
        self.att_head = att_head
        self.head_size = head_size
        self.dense_size = dense_size
        self.output_size = output_size
        
        self.layer_1_nm = tf.keras.layers.LayerNormalization()
        self.layer_2_att = tf.keras.layers.MultiHeadAttention(att_head,head_size )
        self.layer_3_nm = tf.keras.layers.LayerNormalization()
        self.layer_4_dense = tf.keras.layers.Dense(dense_size, activation = 'relu')
        self.layer_5_dense = tf.keras.layers.Dense(output_size)
    
    def call(self, inputs, mask=None):
        
        """
        Takes the input and the mask if it's causal attention
        """
        
        layer_1 = self.layer_1_nm(inputs)
        layer_2 = self.layer_2_att(query = layer_1, key = layer_1, value = layer_1, attention_mask = mask)+layer_1
        layer_3 = self.layer_3_nm(layer_2)
        layer_4 = self.layer_4_dense(layer_3)
        layer_5 = self.layer_5_dense(layer_4)+ layer_3
             
        return layer_5

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "att_head":self.att_head,
            "head_size":self.head_size,
            "dense_size":self.dense_size,
            "output_size": self.output_size
        })
        return config
        

