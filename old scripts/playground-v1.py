# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:38:15 2024
link:
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/fmri_usage.ipynb
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/working_with_rois.ipynb

@author: YSK
"""
import os, time
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thingsmri.dataset import ThingsmriLoader

from nilearn.masking import unmask



dl = ThingsmriLoader(
    thingsmri_dir='.'
)

# ID of subject to analyze
# loading the data of subject 1 takes ~13 min 
sub = '01'
start_time = time.time()
responses, stimdata, voxdata = dl.load_responses(sub)
laoding_time = (time.time() - start_time)/60
print(f"--- {laoding_time:.1f} minutes ---" )


basedir = os.getcwd()
betas_csv_dir = pjoin(basedir, 'betas_csv')

'''
# One ROI at a time
ROI_select = 'rLOC'

# We define a mask of all voxels that match this ROI and apply it to the response data
roimask = voxdata[ROI_select].values.astype(bool)

# each row contains a voxel in the target ROI
roidata = responses[roimask]

# the mean LOC response over the first 100 images
plt.plot(range(100), roidata.mean(axis=0).to_numpy()[:100])
plt.xlabel('trial number');plt.ylabel('mean lLOC amplitude')
plt.show()
'''

LOC_rois = ['rLOC', 'lLOC']
roimask = voxdata[LOC_rois].sum(axis=1).values.astype(bool)
roidata = responses[roimask]

stim_f = pjoin(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
stimdata = pd.read_csv(stim_f)

#parkinsight unit objects/animals
concepts = ['ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf', 'ant', 'apple', 'axe', 'banana', 'bear', 'bee', 'bell', 'boat', 'bottle', 'brush', 'bus', 'butterfly', 'cactus', 'cat', 'chair', 'computer', 'couch', 'cow', 'cup', 'dog', 'dolphin', 'dress', 'elephant', 'faucet', 'fish', 'flamingo', 'fork', 'fox', 'frog', 'giraffe', 'guitar', 'hammer', 'hat', 'horse', 'kangaroo', 'kite', 'knife', 'ladder', 'lion', 'megaphone', 'microscope', 'monkey', 'motorcycle', 'mushroom', 'pencil', 'penguin', 'piano', 'popcorn', 'rabbit', 'rhinoceros', 'rifle', 'scissors', 'scooter', 'scorpion', 'sheep', 'shoe', 'spider', 'spoon', 'sunflower', 'table', 'tie', 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'whale', 'wolf']
count_null = 0

for c in concepts:
    plt.clf()
    print(f"...analyzing concept:{c}")
    query = f'concept == "{c}"'
    
    con_indices = stimdata.query(query).index
    if con_indices.any():
        con_responses = responses[con_indices]
        plt.plot(range(len(con_indices)), roidata.iloc[1, con_indices].to_numpy(), c='r')
        plt.plot(range(len(con_indices)), roidata.iloc[2, con_indices].to_numpy(), c='g')
        plt.legend(["LOC voxel 1", "LOC voxel 2"], loc='lower right')
        plt.xlabel(f'trials');
        plt.ylabel('voxel amplitude')
        plt.title(f"LOC activity for '{c}'")
        plt.savefig(f'figures/subject-{sub}_concept-{c}', dpi=500)
    else:
        print(f"-{c} was not fount...!")
        count_null+=1



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






