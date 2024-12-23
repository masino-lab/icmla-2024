# %% imports
import numpy as np
import os
import csv

# %% constants
SEED = 654321
dir_data_root = "../../data"

# %% defs
def convert_freq_to_float(freq):
    """Converts HPO frequency terms to float values."""
    if freq is None:
        return None
    if '/' in freq:
        num = freq.split('/')[0]
        den = freq.split('/')[1]
        return float(num) / float(den)
    elif '%' in freq:
        return float(freq.replace('%', '')) / 100
    else:
        match freq:
            case 'HP:0040280':
                return 1
            case 'HP:0040281':
                return (99+80)/200
            case 'HP:0040282':
                return (30+79)/200
            case 'HP:0040283':
                return (29+5)/200
            case 'HP:0040284':
                return 2.5/100
            case 'HP:0040285':
                return 0.0
            case _:
                return None
    return freq

# %% load nodes
hpo_2_idx = {} # map hpo concept_id to index
idx_2_hpo = {} # map index to hpo concept_id
hpo_concepts = []
with open(os.path.join(dir_data_root, 'processed', 'hpo_class_nodes.csv'), 'r') as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        node_idx = int(row['node_idx'])
        hpo_id = row['hpo_id']
        hpo_2_idx[hpo_id] = node_idx
        idx_2_hpo[node_idx] = hpo_id
        hpo_concepts.append(hpo_id)

neighbours = {} # map node index to list of neighbouring node indices
with open(os.path.join(dir_data_root, 'processed', 'hpo_is_a_edges.csv'), 'r') as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        src = int(row['source_id'])
        dest = int(row['destination_id'])
        if src not in neighbours:
            neighbours[src] = []
        if dest not in neighbours:
            neighbours[dest] = []
        neighbours[src].append(dest)
        neighbours[dest].append(src)

# %% load gene to phenotype data
file = os.path.join(dir_data_root, 'external', 'hpo', 'genes_to_phenotype.txt')
gene_dict = {}
with open(file, 'r') as in_file:
    reader = csv.DictReader(in_file, delimiter='\t')
    for row in reader:
        ncbi_id = row['ncbi_gene_id']
        gene_symbol = row['gene_symbol'] if row['gene_symbol'] != '-' else None
        hpo_id = row['hpo_id'].replace(':', '_')
        frequency = convert_freq_to_float(row['frequency'])
        disease_id = row['disease_id']
        if ncbi_id not in gene_dict:
            gene_dict[ncbi_id] = {'hpo_concepts': [], "hpo_indices":[], 'disease_ids': [], 'frequencies': [], 'gene_symbol': gene_symbol}
        gene_dict[ncbi_id]['hpo_concepts'].append(hpo_id)
        gene_dict[ncbi_id]['hpo_indices'].append(hpo_2_idx[hpo_id])
        gene_dict[ncbi_id]['disease_ids'].append(disease_id)
        gene_dict[ncbi_id]['frequencies'].append(frequency)

def get_k_hop_neighbours(node, k, neighbours):
    if k == 0:
        return [node]
    if node not in neighbours:
        return []
    result = []
    for n in neighbours[node]:
        result.extend(get_k_hop_neighbours(n, k-1, neighbours))
    return result

def simulate_patient(dict, hop_probability=[.2, .7, 1]):
    hpo_terms = []
    for hpo_concept, hpo_index, freq in zip(dict['hpo_concepts'], dict['hpo_indices'], dict['frequencies']):
        if (freq is None or freq > np.random.rand()) and hpo_concept in hpo_2_idx:
            hop = np.random.rand()
            for i,p in enumerate(hop_probability):
                if hop < p:
                    hop_neighbors = [idx_2_hpo[_] for _ in get_k_hop_neighbours(hpo_index, i, neighbours)]
                    hpo_terms.append(np.random.choice(hop_neighbors))
                    break
    return hpo_terms

def select_hpo_terms(hpo_terms, num_terms):
    return np.random.choice(hpo_terms, num_terms, replace=False)

def simulate_cohort(num_patients, dict, hop_probability=[.2, .7, 1], noise_ratio=0, max_iter=1000):
    cohort = []
    cnt = 0
    iter = 0
    while cnt < num_patients and iter < max_iter:
        iter += 1
        disease_terms = simulate_patient(dict, hop_probability)
        if len(disease_terms) > 0:
            if noise_ratio > 0:
                num_noise_terms = max(int(len(disease_terms) * noise_ratio),1)
                noise_terms = select_hpo_terms(hpo_concepts, num_noise_terms)
                disease_terms.extend(noise_terms)
                cohort.append(list(set(disease_terms)))
            else:
                cohort.append(disease_terms)
            cnt += 1
    return cohort

# %% create simulated patient data assuming single gene disorders
# create 10 simulated patients per gene

dir_out_root = os.path.join(dir_data_root, 'processed', 'simulated_patients_gene_pheno')
# mkdir if not exists
if not os.path.exists(dir_out_root):
    os.makedirs(dir_out_root)

cohort_size = 1000
sim_dict = {
    'optimal': {'hop_probability': [1], 'noise_ratio': 0},
    'imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0},
    'imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0},
    'noise_01': {'hop_probability': [1], 'noise_ratio': 0.1},
    'noise_02': {'hop_probability': [1], 'noise_ratio': 0.5},
    'noise_imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.1},
    'noise_imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.1},
    'noise_imprecision_03': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.5},
    'noise_imprecision_04': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.5},
}
max_iter = 1000

# test code
cohort_size = 50
sim_dict = {'optimal': {'hop_probability': [1], 'noise_ratio': 0},
    'imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0},
    'noise_01': {'hop_probability': [1], 'noise_ratio': 0.1},
    'noise_imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.1},
}
g_cnt = 0

for k,v in sim_dict.items():
    out_dir = f"{k}_hop_prob_{''.join([str(_)+'_' for _ in v['hop_probability']])}noise_ratio_{v['noise_ratio']}"
    out_dir = os.path.join(dir_out_root, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    g_cnt = 0
    for g, d in gene_dict.items():
        out_file = os.path.join(out_dir, f"{g}.csv")
        cohort = simulate_cohort(cohort_size, d, v['hop_probability'], v['noise_ratio'], max_iter)
        with open(out_file, 'w+') as out_file:
            for patient in cohort:
                out_file.write(','.join(patient) + '\n')
        g_cnt += 1
        if g_cnt == 2:
            break
# %%
