########### Standard Packages
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import time
import importlib
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime


########### Packages for plotting
import matplotlib as mpl
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 18


########### Our code
from useful_functions import *
import resource_bifiltration as RB

###################### SET PARAMS FOR PARTICULAR RUN ######################
thresh_type = 'count'
len_thresh = 1000

##################################################################
path_to_files = 'geographic_data/'
results_folder = 'Results_Parks_Resource'

now = datetime.now()
strdate = now.strftime("%Y-%m-%d")

########## SET UP PARAMETERS ##########
print('Setting up params ...')

qs = np.arange(0, 31, 6)

rs = np.arange(2000, 999, -200)

score_col = 'acres'

all_params = {
            'rs': rs, 
            'qs': qs, 
            'quality score col': score_col, 
            'threshold type': thresh_type, 
            'threshold': len_thresh,
            }
                

fileextra = f'_{thresh_type}_{str(len_thresh).replace('.', 'p')}'

print('Loading data ...')
########## Load in Data ##########
blocks = gpd.read_file(path_to_files + 'chicago_blocks.shp')
parks = gpd.read_file(path_to_files + 'parks.shp')

D = np.load(path_to_files + 'parks_distances_l1.npy')

########## Make adjacency graph and set up RB ##########
G = adjacency_graph(blocks, calc_area=True)
rb = RB.ResourceBifiltration(scores = parks[score_col].values, D = D, G = G)


########## Make sure folders exist ##########
os.makedirs(f'{results_folder}/{strdate}', exist_ok=True)
results_folder = results_folder + f"/{strdate}/"


########## RUN AND SAVE CODE ##########
# Save log with outputs from running TrackedComponentCollection
with open(f'{results_folder}/PARKS_LOG{fileextra}.txt', 'w') as f:
    with redirect_stdout(f):
        print("Running code to generate TCC.")
        print(f"r values: {rs}")
        print(f"q values: {qs}")
        print("Code started at:", now.strftime("%Y-%m-%d_%H:%M:%S"))
        TCC = RB.TrackedComponentCollection(rb, rs, qs, len_thresh = len_thresh, blocks = blocks, thresh_type=thresh_type)

# Save TCC produced from TrackedComponentCollection
with open(f'{results_folder}/PARKS_TCC{fileextra}.pkl', 'wb') as file:
    pickle.dump(TCC, file)

# Save params used for this run for reproducibility
with open(f'{results_folder}/PARKS_PARAMS{fileextra}.pkl', 'wb') as file:
    pickle.dump(all_params, file)

    
print("Starting postprocess merging...")
########## Postprocess Merging ##########
G = construct_graph(TCC)

# Components that need to be merged
big_Cs = [C for C in nx.connected_components(G) if len(C) > 1]

# Components that don't need to be merged but save for later
small_cs = [C for C in nx.connected_components(G) if len(C) == 1]
small_TCs = [next(iter(i)) for i in small_cs]

# merge all components that are in the connected component 
merged_TCs = []

# Loop through the components C of the graph G. Each of these is a set of tracked components we want to merge
for i, C in enumerate(big_Cs):
    # print(f"Set {i}")
    # Initialize merged_TC to the first TC in C
    # print("Initializing with TC #0")
    C = list(C)
    merged_TC = C[0]
    
    # Merge in the others one by one
    for j, TC in enumerate(C[1:]):
        # print(f"Merging in TC #{j + 1}")
        merged_TC = TCC.get_merged_tracked_comp(merged_TC, TC)
    merged_TCs.append(merged_TC)
    # print("------------------------------------\n")


all_TCs = merged_TCs+small_TCs



# save for the purpose of making figs, etc.    
with open(f'{results_folder}/PARKS_post_initial_merge{fileextra}.pkl', 'wb') as file:
    pickle.dump(all_TCs, file)