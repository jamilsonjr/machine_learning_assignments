# -*- coding: utf-8 -*-

"""
Created on Fri Oct 30 18:15:04 2020

@author: ist

Usage: visualize_activations(CNN_model,[0,2],test_image)


"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import keras

def visualize_activations(conv_model,layer_idx,image):
    plt.figure(0)
    plt.imshow(image,cmap='gray')
    outputs = [conv_model.layers[i].output for i in layer_idx]
    
    visual = keras.Model(inputs = conv_model.inputs, outputs = outputs)
    
    features = visual.predict(np.expand_dims(np.expand_dims(image,0),3))  
        
    f = 1
    for fmap in features:
            square = int(np.round(np.sqrt(fmap.shape[3])))
            plt.figure(f)
            for ix in range(fmap.shape[3]):
                 plt.subplot(square, square, ix+1)
                 plt.imshow(fmap[0,:, :, ix], cmap='gray')
            plt.show()
            plt.pause(2)
            f +=1
# %%
