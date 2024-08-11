# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:38:15 2024

Tutorials on THINGS dataset:
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/fmri_usage.ipynb
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/working_with_rois.ipynb

To do:
1. test how many concepts exist in the NSD dataset
2. rerun the analysis with NSD 
@author: YSK
"""
import os, time
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial

from thingsmri.dataset import ThingsmriLoader

from nilearn.masking import unmask

# ID of subject to analyze
# loading the data of a single subject takes ~10 minutes
sub = '03'

dl = ThingsmriLoader(
    thingsmri_dir='.'
)
print(f'loading subject: {sub}...')
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

#"C:\Users\User\Projects\THINGS-data"

#parkinsight unit objects/animals
concepts = ['ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf', 'ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'cat', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf']
count_null = 0
count_exist = 0
null = []
df = pd.DataFrame()

#we have 60 unique concepts 
for c in set(concepts):
    plt.clf()
    print(f"--'{c}'")
    query = f'concept == "{c}"'
    con_indices = stimdata.query(query).index

    if con_indices.any():
        print("---found! plotting...")
        count_exist+=1
        df[c] = roidata[con_indices].mean(axis=1)
        #con_responses = responses[con_indices]

        plt.plot(range(len(df[c])), df[c].to_numpy(), c='k')
        plt.xlabel(f'voxels');
        plt.ylabel('voxel amplitude')
        plt.title(f"Avg. LOC activity for '{c}' over {len(con_indices)} trials")
        plt.savefig(f'figures/subject-{sub}_concept-{c}', dpi=500)
    else:
        print(f"---{c} was not fount...!")
        null.append(c)
        count_null+=1
                
print(f"--out of {len(set(concepts))}, {count_exist} exist, {count_null} not:\n---list of not: {null}")

df.to_csv(f"sub{sub}_voxels.csv")


mix_concepts = pd.read_csv('concepts.csv')
ex_concepts = df.columns.values

results = []
for i, c in enumerate(mix_concepts['Concepts']):
    print(f"--similarity: '{c}'")
    c1, c2 = mix_concepts.iloc[i, 2], mix_concepts.iloc[i, 3]
    if c1 in ex_concepts and c2 in ex_concepts:
        sim = 1 - spatial.distance.cosine(df[c1], df[c2])
        results.append([c, sim])
    else:
        print(f"---not found!")
        
pd.DataFrame(results).to_csv(f"sub{sub}_similarity.csv")






