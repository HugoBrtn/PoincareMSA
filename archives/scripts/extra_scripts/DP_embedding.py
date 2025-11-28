import sys
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn.functional as F

#import torchbearer
#from torchbearer.callbacks import EarlyStopping

import seaborn as sns

#from torchinfo import summary 


# Directory containing STP embeddings
class1_dir = 'embeddings/ankh_base_globins'
class1_dir = 'embeddings/ankh_base_globins'

# Directory containing non-STP embeddings
class2_dir ='embeddings/ankh_base_globins'
class2_dir = 'embeddings/ankh_base_globins'
# Directory to save padded embeddings and tensors
#output_dir = "/dsimb/rennes/ghallab/Documents/PREP_DONNEES/SORTIES_MMSEQS/EMBEDDINGS/padded_embeddings"

def pad_embeddings(embedding):
    padding_length = max_length - len(embedding)
    padded_embedding = torch.cat((embedding, torch.zeros(padding_length, embedding.shape[1])), dim=0)
    return padded_embedding

# Read class 1 embeddings | STP
class1_embeddings = []
for filename in os.listdir(class1_dir):
    embedding = torch.load(os.path.join(class1_dir, filename))
    class1_embeddings.append(embedding)
    
# Read class 2 embeddings - NON STP 
class2_embeddings = []
for filename in os.listdir(class2_dir):
    embedding = torch.load(os.path.join(class2_dir, filename))
    class2_embeddings.append(embedding)
    
# Pad embeddings to a fixed length
# embedding['representations'][5] is in fact the tensor comprise in the dictionnary "embedding".
max_length = 768

print(embedding['embedding'].shape, file=sys.stderr)
print(embedding['embedding'], file=sys.stderr)


#for j in range(0, 8):
#    tot_value=0
#    for i in range(0, len(embedding['representations'][33])):
#        tot_value=embedding['representations'][33][i][j] 
    
                      

# Padded embedding['representations'][5] is in fact the tensor comprise in the dictionnary "embedding".
#class1_padded_embeddings = [pad_embeddings(embedding['representations'][33]) for embedding in class1_embeddings]
#class2_padded_embeddings = [pad_embeddings(embedding['representations'][33]) for embedding in class2_embeddings]

# Unpaded
class1_padded_embeddings = [embedding['embedding'] for embedding in class1_embeddings]
class2_padded_embeddings = [embedding['embedding'] for embedding in class2_embeddings]
print(f'Shape: Class1 {len(class1_padded_embeddings)}',      file=sys.stderr)
print(f'Shape: Class2 {len(class2_padded_embeddings)}',      file=sys.stderr)

#Parametres---------------------------------------------------------------------------------------------------------------------------------

def scoring_matrix(emb1, emb2): #calcul du dot product
    """
    Cosine similarity between embedding 1 and embedding 2 ==> SCORE

    parameters : 
    emb 1 : embedding of sequence 1
    emb 2 : embedding of sequence 2 
    """

    # cosine similarity = normalize the vectors & multiply
    matrix_score = F.normalize(emb1) @ F.normalize(emb2).t()

    #Convert matrix tensor in numpy
    matrix_score_np = matrix_score.numpy()

    return(matrix_score_np)


class ScoreParams:
	'''
	Define scores for each parameter
	'''
	def __init__(self,gap):
		self.gap = gap
		#self.match = match
		#self.mismatch = mismatch

	#def misMatchChar(self,x,y):
	#	if x != y:
	#		return self.mismatch
	#	else:
	#		return self.match

def getMatrix(sizeX,sizeY,gap):
	'''
	Create an initial matrix of zeros, such that its len(x) x len(y)
	'''
	matrix = []
	for i in range(len(sizeX)+1):
		subMatrix = []
		for j in range(len(sizeY)+1):
			subMatrix.append(0)
		matrix.append(subMatrix)

	# Initializing the first row and first column with the gap values
	for j in range(1,len(sizeY)+1):
		matrix[0][j] = j*gap
	for i in range(1,len(sizeX)+1):
		matrix[i][0] = i*gap
	return matrix

def getTraceBackMatrix(sizeX,sizeY):
	'''
	Create an initial matrix of zeros, such that its len(x) x len(y)
	'''
	matrix = []
	for i in range(len(sizeX)+1):
		subMatrix = []
		for j in range(len(sizeY)+1):
			subMatrix.append('0')
		matrix.append(subMatrix)

	# Initializing the first row and first column with the up or left values
	for j in range(1,len(sizeY)+1):
		matrix[0][j] = 'left'
	for i in range(1,len(sizeX)+1):
		matrix[i][0] = 'up'
	matrix[0][0] = 'done'
	return matrix


def globalAlign(x,y,score_matrix):
    '''
    Fill in the matrix with alignment scores
    '''
    score_gap=-1
    matrix    = getMatrix(x,y,score_gap)
    traceBack = getTraceBackMatrix(x,y)


    for i in range(1,len(x)+1):
        for j in range(1,len(y)+1):
            left = matrix[i][j-1] + score_gap
            up   = matrix[i-1][j] + score_gap
            diag = matrix[i-1][j-1] + score_matrix[i-1][j-1]
            #diag = matrix[i-1][j-1] + score_matrix[i-1][j-1]
            matrix[i][j] = max(left,up,diag)
            if matrix[i][j] == left:
                traceBack[i][j] = 'left'
            elif matrix[i][j] == up:
                traceBack[i][j] = 'up'
            else:
                traceBack[i][j] = 'diag'
            #print(f'matrix[i][j]:{matrix[i][j]}') 
    return matrix[i][j],matrix,traceBack

def getAlignedSequences(x,y,matrix,traceBack):
    '''
    Obtain x and y globally aligned sequence arrays using the bottom-up approach
    '''
    xSeq = []
    ySeq = []
    i = len(x)
    j = len(y)
    while(i > 0 or j > 0):
            if traceBack[i][j] == 'diag':
                    # Diag is scored when x[i-1] == y[j-1]
                    xSeq.append(x[i-1])
                    ySeq.append(y[j-1])
                    i = i-1
                    j = j-1
            elif traceBack[i][j] == 'left':
                    # Left holds true when '-' is added from x string and y[j-1] from y string
                    xSeq.append('-')
                    ySeq.append(y[j-1])
                    j = j-1
            elif traceBack[i][j] == 'up':
                    # Up holds true when '-' is added from y string and x[j-1] from x string
                    xSeq.append(x[i-1])
                    ySeq.append('-')
                    i = i-1
            elif traceBack[i][j] == 'done':
                    # Break condition when we reach the [0,0] cell of traceback matrix
                    break
    return xSeq,ySeq

def printMatrix(matrix):
    '''
    Create a custom function to print the matrix
    '''
    for i in range(len(matrix)):
            print(matrix[i])
    print()

'''
Driver Code:
'''

def align_embed(embedding1,embedding2):
    x = range(1,embedding1.shape[0])
    y = range(1,embedding2.shape[0])

    #print('Input sequences are: ')
    #print(x)
    #print(y)
    #print()
    score = ScoreParams(-1)
    score_matrix=scoring_matrix(embedding1,embedding2)
   
    max_score,matrix,traceBack = globalAlign(x,y,score_matrix)

    #print(f'Max score:{max_score}')

    #print('Printing the score matrix:')
    #printMatrix(matrix)

    #print('Printing the trace back matrix:')
    #printMatrix(traceBack)

    #xSeq,ySeq = getAlignedSequences(x,y,matrix,traceBack)

    #print('The globally aligned sequences are:')
    #print(*xSeq[::-1])
    #print(*ySeq[::-1])
    return max_score,score_matrix

#####################

#Element to pad
#size=10

#num_file=0;

#list files: to modify 
class_padded_embeddings=class1_padded_embeddings

# print(f'id',end='')
# for i in range(0,len(class_padded_embeddings)):
#     print(f'{i}',end='')
#     if (i == (len(class_padded_embeddings)-1)):    
#         print("\n",end='')
#     else:
#         print(',',end='')

   
for n, embedding1 in enumerate(class_padded_embeddings):
    #print(f'File:{n} Shape: {embedding1.shape}', file=sys.stderr)
    #print(f'Shape j: {embedding1.shape[0]} Shape: k {embedding1.shape[1]}', file=sys.stderr)
    embedding1=torch.tensor(embedding1)
    print(f'Embedding id : {n}')
    print(f'Embedding_shape : {embedding1.shape}')
    for m, embedding2 in enumerate(class_padded_embeddings):
        #print(f'File:{m} Shape: {embedding2.shape}', file=sys.stderr)
        #print(f'Shape j: {embedding2.shape[0]} Shape: k {embedding2.shape[1]}', file=sys.stderr)
        embedding2=torch.tensor(embedding2)
        max_score,score_matrix=align_embed(embedding1,embedding2)

        name_file=f'{m}_{n}_scoring_matrix.csv'
        # Save np matrix
        np.savetxt(f'scoring_matrices/{name_file}', score_matrix, delimiter=',')
        print(score_matrix.shape)

        print(f'{max_score}',end='')
        if (m == (len(class_padded_embeddings)-1)):    
            print("\n",end='')
        else:
            print(',',end='')
    break

###################################@

