# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:38:15 2024

Rerun the analysis for all subjects! something is weird
now it is dissimilarity! 

Tutorials on THINGS dataset:
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/fmri_usage.ipynb
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/working_with_rois.ipynb

@author: YSK
"""
import time
from os.path import join as pjoin
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, correlation
from scipy.stats import permutation_test, spearmanr

from thingsmri.dataset import ThingsmriLoader

#"C:\Users\User\Projects\THINGS-data"

# ID of subject to analyze
## loading the data of a single subject takes ~10 minutes
sub = '03'

dl = ThingsmriLoader(
    thingsmri_dir='.'
)

print(f'Loading subject number: {sub}...')
start_time = time.time()
responses, stimdata, voxdata = dl.load_responses(sub)
loading_time = (time.time() - start_time)/60
print(f"-loaded! took {loading_time:.1f} minutes ---" )

'''
1) responses (211339 voxels, 9840 trials) 
2) stimdata (9840 trials, 7 parameters)
3) voxdata (211339 voxels, 222 parameters)
'''    

LOC_rois = ['rLOC', 'lLOC']

# selecting all columns in the voxel metadata
# and sum over them to create a boolean mask that reflects their union
roimask = voxdata[LOC_rois].sum(axis=1).values.astype(bool)
# applying this mask to get the activation of LOC voxels to each image
roidata = responses[roimask]
print(f"-in THINGS there are {len(set(stimdata.concept))} concepts")
# 720 concepts 

#parkinsight unit objects/animals
cs = pd.read_csv("concepts - pilot.csv")
chimerical = cs["Concepts"]
concepts = [y for x in [cs["concept 1"], cs["concept 2"]] for y in x]

print(f"--loaded data for {len(set(concepts))} single concepts that form the {len(chimerical)} chimerical animals")
# 44 unqiue concepts out of 94 chimerical animals
#print(set(concepts))
## {'handbag', 'squirrel', 'eagle', 'zebra', 'kangaroo', 'desk', 'hat', 'bottle', 'turtle', 'bee', 'chair', 'chicken', 'spray', 'sheep', 'knife', 'owl', 'heart', 'horse', 'monkey', 'donkey', 'duck', 'star', 'windmill', 'swan', 'elephant', 'gorilla', 'bus', 'peacock', 'couch', 'alligator', 'camel', 'fish', 'hammer', 'scissors', 'guitar', 'shoe', 'mouse', 'cow', 'cat', 'penguin', 'rooster', 'banana', 'bear', 'brush', 'checken', 'rhinoceros', 'megaphone', 'bird', 'raccoon', 'ant', 'microscope', 'seal', 'kite', 'sunflower', 'rabbit', 'fork', 'faucet', 'trumpet', 'elephant ', 'umbrella', 'ostrich', 'dog', 'axe', 'piano', 'shaver', 'frog', 'fox', 'tiger', 'leopard', 'scooter', 'boat', 'goat', 'tea', 'telephone', 'deer', 'lion', 'giraffe'}

count_exist = 0
null = []
df = pd.DataFrame()

for c in set(concepts):
    plt.clf()
    print(f"--'{c}'")
    query = f'concept == "{c}"'
    con_indices = stimdata.query(query).index

    if con_indices.any():
        count_exist+=1
        df[c] = roidata[con_indices].mean(axis=1)
        #con_responses = responses[con_indices]
        #print("---found! plotting...")
        plt.plot(range(len(df[c])), df[c].to_numpy(), c='k')

        #plt.plot(range(len(con_indices)), roidata.iloc[1, con_indices].to_numpy(), c='r')
        #plt.plot(range(len(con_indices)), roidata.iloc[2, con_indices].to_numpy(), c='g')
        #plt.legend(["LOC voxel 1", "LOC voxel 2"], loc='lower right')
        plt.xlabel(f'voxels');
        plt.ylabel('voxel amplitude')
        plt.title(f"Avg. LOC activity for '{c}' over {len(con_indices)} trials")
        plt.savefig(f'figures/subject-{sub}_concept-{c}', dpi=500)
    else:
        #print("---not found---")
        null.append(c)
                
#print(f"-- {count_exist} out of {len(set(concepts))} single concepts exist")
# 30 out of 44 exist 

print(set(null))
#sub01-{'ostrich', 'elephant ', 'duck', 'chicken', 'peacock', 'checken', 'seal', 'mouse', 'gorilla', 'eagle', 'rooster', 'owl', 'swan', 'fox'}
#sub02-{'ostrich', 'rooster', 'mouse', 'swan', 'owl', 'peacock', 'duck', 'seal', 'eagle', 'elephant ', 'gorilla', 'chicken', 'checken', 'fox'}
#sub03-{'rooster', 'mouse', 'elephant ', 'peacock', 'duck', 'eagle', 'swan', 'checken', 'seal', 'chicken', 'fox', 'ostrich', 'owl', 'gorilla'}
df.to_csv(f"sub{sub}_concepts_voxels.csv")

#part B - running similarity analysis
#sub = '01'
#cs = pd.read_csv("concepts - pilot.csv")
#df = pd.read_csv(f"sub{sub}_concepts_voxels.csv")

# Custom function for cosine distance without the axis argument
def cosine_statistic(x, y):
    return cosine(x, y)

def correlation_statistic(x, y):
    return correlation(x, y)

mix_concepts = cs
ex_concepts = df.columns.values[1:]

results = []
for i, c in enumerate(mix_concepts['Concepts']):
    c1, c2 = mix_concepts.iloc[i, 2], mix_concepts.iloc[i, 3]
    if c1 in ex_concepts and c2 in ex_concepts:
        res_cos = permutation_test(data=(df[c1], df[c2]), 
                                   statistic=cosine_statistic, 
                                   vectorized=False, 
                                   n_resamples=10000, 
                                   alternative='two-sided')
        
        res_corr = permutation_test(data=(df[c1], df[c2]), 
                                    statistic = correlation_statistic, 
                                    vectorized=False,
                                    n_resamples=10000, 
                                    alternative='two-sided')
        
        results.append([c, res_cos.statistic, res_corr.statistic])
    #else:
    #    print(f"---{c} not found!")

print(f"--{len(results)} joint concepts have fMRI data!")
## 50 matching concepts have both data!
pd.DataFrame(results, columns=['Concept', 'Cosine Similarity', 'Correlation']).to_csv(f"sub{sub}_similarity - pilot.csv")

rs = pd.read_csv(f"sub{sub}_similarity - pilot.csv")

def spearmanr_statistic(x, y):
    return spearmanr(x, y)[0]

cs_filtered = cs.loc[cs['Concepts'].isin(rs["Concept"])]
   
corr_tests = ['avg-real','avg-conf', 'log-phy','clip-real','clip-vis-sim']
for t in corr_tests:
    res_cos = permutation_test(data=(cs_filtered[t], rs["Cosine Similarity"]), 
                                statistic = spearmanr_statistic, 
                                vectorized=False,
                                n_resamples=10000, 
                                alternative='two-sided')
    print(f"{t} and cosine:\n --r = {res_cos.statistic:.3f}, p = {res_cos.pvalue:.5f}") 

for t in corr_tests:
    res_corr = permutation_test(data=(cs_filtered[t], rs["Correlation"]), 
                                statistic = spearmanr_statistic, 
                                vectorized=False,
                                n_resamples=10000, 
                                alternative='two-sided')
    print(f"{t} and corr:\n --r = {res_corr.statistic:.3f}, p = {res_corr.pvalue:.5f}") 

#sub01:
    ## avg-real~cosine: r = 0.312, p = 0.03400
    ## clip-real~cosine: r = 0.401, p = 0.00620
    ## clip-real~corr  : r = 0.373, p = 0.00860
    
#sub02:
    ## nothing significant
    
#sub03:
    ## clip-real and cosine: r = -0.371, p = 0.01060
    ## clip-real and corr: r = -0.315, p = 0.03060


'''
loading only stim data:
    
    #basedir = os.getcwd()
    #betas_csv_dir = pjoin(basedir, 'betas_csv')
    #stim_f = pjoin(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
    #stimdata = pd.read_csv(stim_f)
    
#three types of trials: train, test and catch. The latter with non-object
'''






