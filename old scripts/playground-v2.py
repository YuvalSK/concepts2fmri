# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:38:15 2024
Tutorials on THINGS dataset:
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/fmri_usage.ipynb
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/working_with_rois.ipynb

@author: YSK
"""
import os, time
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

from thingsmri.dataset import ThingsmriLoader

from nilearn.masking import unmask

# ID of subject to analyze
# loading the data of subject 01 took ~12 min 
# loading the data of subject 02 took ~11 min 
# loading the data of subject 03 took ~5 min 

sub = '03'

dl = ThingsmriLoader(
    thingsmri_dir='.'
)
print(f'loading data from subject: {sub}...')
start_time = time.time()
responses, stimdata, voxdata = dl.load_responses(sub)
loading_time = (time.time() - start_time)/60
print(f"data loaded! it took {loading_time:.1f} minutes ---" )
    
#"C:\Users\User\Projects\THINGS-data"

LOC_rois = ['rLOC', 'lLOC']
roimask = voxdata[LOC_rois].sum(axis=1).values.astype(bool)
roidata = responses[roimask]

basedir = os.getcwd()
betas_csv_dir = pjoin(basedir, 'betas_csv')
stim_f = pjoin(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
stimdata = pd.read_csv(stim_f)

#parkinsight unit objects/animals
concepts = ['ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf', 'ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'cat', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf']
count_null = 0
count_exist = 0
null = []
df = pd.DataFrame()

for c in concepts:
    plt.clf()
    print(f"-analyzing concept:{c}")
    query = f'concept == "{c}"'
    con_indices = stimdata.query(query).index

    if con_indices.any():
        count_exist+=1
        df[c] = roidata[con_indices].mean(axis=1)
        #con_responses = responses[con_indices]
        print("--found! plotting...")
        plt.plot(range(len(df[c])), df[c].to_numpy(), c='k')

        #plt.plot(range(len(con_indices)), roidata.iloc[1, con_indices].to_numpy(), c='r')
        #plt.plot(range(len(con_indices)), roidata.iloc[2, con_indices].to_numpy(), c='g')
        #plt.legend(["LOC voxel 1", "LOC voxel 2"], loc='lower right')
        plt.xlabel(f'voxels');
        plt.ylabel('voxel amplitude')
        plt.title(f"Avg. LOC activity for '{c}' over {len(con_indices)} trials")
        plt.savefig(f'figures/subject-{sub}_concept-{c}', dpi=500)
    else:
        print(f"--{c} was not fount...!")
        null.append(c)
        count_null+=1
                

df.to_csv(f"sub{sub}_voxels.csv")
#sim = pdist(df.to_numpy(), metric='cosine')

#sim_res = squareform(sim)



'''
#three types of trials: train, test and catch. The latter with non-object

from os.path import join as pjoin
import glob
import numpy as np
import pandas as pd
from nilearn.masking import unmask
from nilearn.plotting import plot_stat_map
from nilearn.image import load_img, index_img
import matplotlib.pyplot as plt
import cortex
import os


basedir = os.getcwd()

sub = '01'

betas_csv_dir = pjoin(basedir, 'betas_csv')
betas_vol_dir = pjoin(basedir, 'betas_vol', f'sub-{sub}')
#path_to_analyze = pjoin(betas_vol_dir, '*', '*')

data_file = pjoin(betas_csv_dir, f'sub-{sub}_ResponseData.h5')
responses = pd.read_hdf(data_file)  # this may take a minute
print('Single trial response data')
responses.head()

vox_f = pjoin(betas_csv_dir, f'sub-{sub}_VoxelMetadata.csv')
voxdata = pd.read_csv(vox_f)
voxdata.head()

print('available voxel metadata:\n', voxdata.columns.to_list())

# Stimulus metadata
stim_f = pjoin(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
stimdata = pd.read_csv(stim_f)
stimdata.head()

#we can select data based on which object category (or concept) was shown - e.g., here all trials with images of mango

mango_indices = stimdata.query('concept == "mango"').index
mango_responses = responses[mango_indices]
mango_responses.shape






