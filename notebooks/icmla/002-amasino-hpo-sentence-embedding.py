# %% imports
from sentence_transformers import SentenceTransformer
import csv
import torch
import os
import numpy as np

print(torch.cuda.is_available())
print(os.getcwd())

# %% load sentence transformer model
model = SentenceTransformer('sentence-t5-large')

# %% encode hpo definitions
file_hpo_nodes = '../../data/processed/hpo_class_nodes.csv'
file_hpo_node_embeddings = '../../data/processed/hpo_class_node_desc_embeddings'

# read the input file and encode the definitions then save the embeddings
cnt = 0
embeddings = []
with open(file_hpo_nodes, 'r') as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        node_idx = int(row['node_idx'])
        if node_idx != cnt:
            print(f'Error: node_idx {node_idx} != {cnt}')
            break
        label = row['label']
        definition = row['definition']
        embeddings.append(model.encode(f'{label} : {definition}'))
        cnt += 1
        if cnt % 500 == 0:
            print(f'Processed {cnt} nodes')
np.save(file_hpo_node_embeddings, np.array(embeddings), allow_pickle=True)
# %%
