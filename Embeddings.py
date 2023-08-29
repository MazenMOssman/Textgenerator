#!/usr/bin/env python
# coding: utf-8

# In[94]:


import tensorflow as tf

class embeddings(tf.keras.layers.Layer):
    
    def __init__(self, embedding_input, embedding_out, inputs_length):
        
        super().__init__()
                
        self.embedding_input = embedding_input
        self.embedding_output = embedding_out
        self.embedding_length = inputs_length
        
        self.token_embed = tf.keras.layers.Embedding(self.embedding_input, self.embedding_output, mask_zero = True)
        self.positional_embed = tf.keras.layers.Embedding(self.embedding_length, self.embedding_output)        
        
    
    
    def call(self, inputs):
        
        token_embed = self.token_embed(inputs)
        
        input_length_embedding = tf.range(start = 0, limit = self.embedding_length, delta = 1)
        
        positional_embed = self.positional_embed(self.embedding_length)
        
        embedding = token_embed + positional_embed
        
        return embedding

    
    def get_config(self):
        
        config = super().get_config()
        config.update({'embedding_input' : self.embedding_input,
                   'embedding_output': self.embedding_output,
                 'inputs_length': self.embedding_length })
        return config
    


# In[ ]:




