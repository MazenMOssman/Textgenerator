#!/usr/bin/env python
# coding: utf-8

# In[87]:


import tensorflow as tf
class data_load():
    """
    Loads, split, and tokenize the dataset.
    
    Parameters:
    Path: The path of the text dataset 
    Returns:train dataset, valid dataset, tokenizer
    """
    def __init__(self, path):
        
        super().__init__()
        self.path = path
        self.tokenizer = tf.keras.layers.TextVectorization(split = 'character', standardize = 'lower')
        self.dataset = None

        
    def __read_data(self):
        
        with open(self.path, 'r') as file:
            self.dataset = file.read()
            
            
    def __window_shaping(self, data ,window_size, batch):
        
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.window(size = window_size + 1, shift = 1, drop_remainder = True )
        data = data.flat_map(lambda window: window.batch(window_size + 1))
        data = data.map(lambda window: (window[:-1],window[1:]))
        data = data.batch(batch)
        return data
            

    
    def __call__(self, windows_size, batch_size, data_split_ratio):
        
        
        """
        Parameters: 
        
        windows_size: the length of the windows
        batch_size: the batch size of the datset
        data_split_ratio: split ratio of validation and train data
        
        Returns:
        
        train dataset, valid dataset, tokenizer
        
        """
        
        self.__read_data()
        train_dataset = self.dataset[:int(len(self.dataset)*data_split_ratio)]
        valid_dataset = self.dataset[int(len(self.dataset)*data_split_ratio):]
        
        self.tokenizer.adapt([train_dataset])
        
        train_dataset = self.tokenizer(train_dataset)-2
        train_dataset = self.__window_shaping(train_dataset, windows_size, batch_size)
        
        valid_dataset = self.tokenizer(valid_dataset)-2
        valid_dataset = self.__window_shaping(valid_dataset, windows_size, batch_size)
        
        return train_dataset, valid_dataset, self.tokenizer
    


# In[ ]:




