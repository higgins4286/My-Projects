#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code is the neural network algorithm for predicting potential energy
coefficients of Silicon from bispectrum coefficients. The framework of this 
code is based off the NN from Yanxon et al. 2020.

Throughout the code I reference A and alpha. They are:
    A     = the potential energy (PE) coefficients
    alpha = the exponential, spherical fcn. coefficients

"""

###Import necessary files
import numpy as np

'''
Step 1.
Prepare the data for the NN 
'''

'''
Step 1.a.

Import data and create a dictionary that holds all atom types from the 
input bispectrum coefficients and output potential energy coefficients (PE) 
where each key holds an array with input size of ______ and output size of 
16 rows and 2 columns.

Array, dictionary, and index names are as follows:
    
type_dict     = the dictionary where each dictionary key is a type of atom 
                that stores groups of the input and output data
keys          = the key names in the type_dict
PE_long_group = an array that includes the groupings of 16 rows of A and alpha
r             = integer value that represents one row 
'''
## INPUT DATA:
## Import data for bispectrum coefficient data files. 

# bi_data =

## OUTPUT DATA:
## Import data for long, medium, and short bond lengths potential energy 
## coefficient data files. 

PE_long = np.genfromtxt('/Users/laurenhiggins/Desktop/Neural Networking Project 5599/Data/gs_scfV-mb-long.dat', 
                        skip_header=3, 
                        usecols = (0, 1),
                        invalid_raise=False) #Ignore the column inconsistencies
#PE_medium = np.genfromtxt('/Users/laurenhiggins/Desktop/Neural Networking Project 5599/Data/gs_scfV-mb-medium.dat', 
#                        skip_header=3, 
#                        usecols = (0, 1),
#                        invalid_raise=False) #Ignore the column inconsistencies
PE_short = np.genfromtxt('/Users/laurenhiggins/Desktop/Neural Networking Project 5599/Data/gs_scfV-mb-short.dat', 
                        skip_header=3, 
                        usecols = (0, 1),
                        invalid_raise=False) #Ignore the column inconsistencies 

## Create the dictionary to store inputs and outputs per type and assign them to 
## keys.

#Create array to store key names
PE_long_keys = []
PE_medium_keys = [] 
PE_short_keys = []
    
for i in range(1, 341):
    PE_long_keys.append('PE_long_type_' + str(i))
    PE_short_keys.append('PE_short_type_' + str(i))

# ## Create a function that will do the above.
# def key_names(start, end, keys,type_name):
#     # The range needs to be the length of the number of key names needed    
#     for i in range(start, end):
#         return keys.append(type_name + str(i))

# keys_long_test = []    
# keys_long_test = key_names(1, 341, keys_long_test, 'long_test')
   
#Create dictionary where each key entry is 16 rows of the imported PE files
for r in range(0, len(PE_long)):
    type_dict = {PE_long_keys[j]: PE_long[16*j:16*(j+1)] for j in range(len(PE_long_keys))}
    type_dict_short = {PE_short_keys[j]: PE_short[16*j:16*(j+1)] for j in range(len(PE_short_keys))}

type_dict.update(type_dict_short)
    
'''
Step 1.b.
Use Normalizer() module from sklearn.
How this works:
    Normalize samples individually to unit norm.
    This process uses gradient descent normalize the input data
    
    Each sample (i.e. each row of the data matrix) with at least one non zero 
    component is rescaled independently of other samples so that its 
    norm (l1, l2 or inf) equals one.
    
    This transformer is able to work both with dense numpy arrays and 
    scipy.sparse matrix (use CSR format if you want to avoid the burden of a 
    copy / conversion).
    
    Scaling inputs to unit norms is a common operation for text classification 
    or clustering for instance. For instance the dot product of two 
    l2-normalized TF-IDF vectors is the cosine similarity of the vectors 
    and is the base similarity metric for the Vector Space Model commonly used 
    by the Information Retrieval community.
    
    Note: 
        
    MLPRegressor trains iteratively since at each time step the partial 
    derivatives of the loss function with respect to the model parameters are 
    computed to update the parameters.

    It can also have a regularization term added to the loss function that 
    shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy 
    arrays of floating point values.
    
    See more info here:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
'''

###Pre-process data
# So far all I think I need to do is Normalize it. Maybe I could also do some
# feature extraction to account for symmetry?

# from sklearn.preprocessing import Normalizer
# scaler = Normalizer()
# #symmetry = preprocess for symmetry same energy? Might be redundant/already 
# # considered in the bispectrum code

# '''
# STOP: This will be used if output error is high.

# Step 1.b. (optional)

# Optimize the distribution of the training/testing sets to be used in the 
# stratify parameter of train_test_split.
# '''

# # Visualize data to find the necessary trends in distribution to optimize
# # the stratify param.

# #Find correlations in the data
# from pandas.plotting import scatter_matrix

# attributes = #important attributes
# scatter_matrix(bi_pe_data[attributes], figsize=(12,8))



# '''
# Step 1.c.

# Split the input/output pairs into a 70/30 training/testing split. I need to
# make these sets representitive of the input distribution, random_state is the 
# variable.
# '''


# from sklearn.model_selection import train_test_split
# bi_train, bi_test, pec_train, pec_test = train_test_split(bi_array, 
#                                                           pec_binary, 
#                                                           test_size=0.3, 
#                                                         train_size=0.7, 
                                                        
# # Random state is a key paramater that controls how the training/testing
# # sets are sampled. The distribution of samples needs to reflect the 
# # distribution of the actual inputs. Use stratify to optimize the distribution 
# # of the data.
#                                                         random_state=66,
#                                                         stratify=None)


# '''
# Step 2.
# NN Framework:
#     input layer
#     hidden layer
#     bias? (add this in if the goodness measure is low)
#     weights
#     output
# '''

# from sklearn.pipeline import make_pipeline
# from sklearn.neural_network import MLPRegressor

# '''
# NN regression using hidden layers and the activation function and solver 
# that was used in the Adam paper.

# regr_adam = MLPRegressor(
#     hidden_layer_sizes = 2 layers with 100 units per layer
#     activation = rectified linear unit (relu) function Activation fcn for 
#                  hidden layer
#     solver = ADAM, stochastic gradient-based optimizer proposed by Kingma, 
#              Diederik, and Jimmy Ba)

# How do I implement bias and control weights?
# '''
# regr_adam = MLPRegressor(hidden_layer_sizes=(100,100),
#                          activation='relu', 
#                          solver='adam')

# #Train NN
# model = regr_adam.fit(bi_train, pec_train)

# ##Store Output values for analysis
# probs = model.predict_proba(bi_test[:2])
# pred_vals = model.predict(bi_test[:2])

# '''
# Save pred_vals to an .csv files to compaire how a change of paramaters 
# change the output. More into on how to do this can be found here:
#     https://www.geeksforgeeks.org/writing-csv-files-in-python/
# '''

# import csv  
      
# # Column headers. I am not quite sure what these will be. Maybe the following  
# headers = [<column names>]  
      
# # Data rows of csv file. These will be the output values from 
# # pred_vals. 
# rows = [ [<column values>],  
#          [<column values>],  
#          [<column values>],  
#          [<column values>],  
#          [<column values>],  
#          [<column values>]]  
      
# # name of csv file  
# filename = "NN_PE_Coefficients"+ <insert version marker here> +".csv"
      
# # writing to csv file  
# with open(filename, 'w') as csvfile:  
#     # creating a csv writer object  
#     csvwriter = csv.writer(csvfile)  
          
#     # writing the fields  
#     csvwriter.writerow(fields)  
          
#     # writing the data rows  
#     csvwriter.writerows(rows)


# ##Goodness measurements

# #Create comparison plots of the known PEC and the results of training and testing.
# import matplotlib.pyplot as plt

# #I want one plot with three bars, each representing training, testing, and true.
# fig1, ax1 = plt.subplots(figsize=[8,6])

# #Each bar will have this width
# width = 0.3

# #Each bar will plot around this point on the x-axis
# x_place = 1

# #Plotting the training set output
# ax1.bar(x_place-width, 
#         sizes_bar1, 
#         width = width, 
#         bottom = None, 
#         align = 'center', 
#         color = 'cyan',
#         label = 'Training'.
#         data = None)

# #Plotting the testing set output
# ax1.bar(x_place, 
#         sizes_bar2, 
#         width = width, 
#         bottom = None, 
#         align = 'center', 
#         color = 'blue',
#         label = 'Testing',
#         data = None)

# #Plotting the true value values
# ax1.bar(x_place+width, 
#         sizes_bar3, 
#         width = width, 
#         bottom = None, 
#         align = 'center', 
#         color = 'gold',
#         label = 'True Values'
#         data = None)

# #Define athestics of the plot
# ax1.set_ylabel('Potential Energy Coefficients (meV/atom)', fontsize=16)
# ax1.set_xlabel('Dataset', fontsize=16)

# #Create legend
# ax.legend(loc='upper left', fontsize=12)
# ax.grid(which='both', axis='both')

# #Save the plot
# #plt.savefig('/Users/laurenhiggins/Desktop/<insert file path here>', bbox='tight')

# plt.show()


