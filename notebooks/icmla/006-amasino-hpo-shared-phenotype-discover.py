# %% imports
import os
import csv
import numpy as np
from numpy.linalg import norm
import pygad
import random
#import polars as pl
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing

# print("Imports completed ...")

# %% set globals
# dir_data_root = "/Users/amasino/dev/clemson-gitlab/population-phenotyping/data"
dir_data_root = "/home/amasino/indigo_amasino_lab/amasino/dev/population-phenotyping/data"
slurm_log = f'/home/amasino/indigo_amasino_lab/amasino/dev/population-phenotyping/slurm/slurmlog.txt'
SEED = 654321

# %% load embeddings
#embeddings = np.load('../../data/processed/hpo_class_node_gnn_embeddings.npy', allow_pickle=True)
embedding_model = f'mpnet'
embeddings = np.load(os.path.join(dir_data_root, 'processed', f'hpo_class_node_desc_embeddings_model_{embedding_model}.npy'), allow_pickle=True)

# print("Embeddings loaded ...")

# %% load node data
concept_2_idx = {}
idx_2_concept = {}
with open(os.path.join(dir_data_root, 'processed', 'hpo_class_nodes.csv'), 'r') as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        node_idx = int(row['node_idx'])
        hpo_id = row['hpo_id']
        concept_2_idx[hpo_id] = node_idx
        idx_2_concept[node_idx] = hpo_id

# %% load gene data
file = os.path.join(dir_data_root, 'external', 'hpo', 'genes_to_phenotype.txt')
gene_dict = {}
with open(file, 'r') as in_file:
    reader = csv.DictReader(in_file, delimiter='\t')
    for row in reader:
        ncbi_id = row['ncbi_gene_id']
        gene_symbol = row['gene_symbol'] if row['gene_symbol'] != '-' else None
        hpo_id = row['hpo_id'].replace(':', '_')
        disease_id = row['disease_id']
        if ncbi_id not in gene_dict:
            gene_dict[ncbi_id] = {'hpo_idx': [], 'disease_ids': [], 'frequencies': [], 'gene_symbol': gene_symbol}
        gene_dict[ncbi_id]['hpo_idx'].append(concept_2_idx[hpo_id])
        gene_dict[ncbi_id]['disease_ids'].append(disease_id)

# %% build node neighbourhs map
neighbours = {}
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

# print("Mappings generated ...")

# %% defs
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def aym_set_similarity(A, B):
    max_sims = []
    Anp = np.array(A)
    Bnp = np.transpose(np.array(B))
    scores = np.matmul(Anp, Bnp)
    scores = np.max(scores, axis=1)
    return np.mean(scores)

def sym_set_similarity(A, B):
    return (aym_set_similarity(A, B) + aym_set_similarity(B, A)) / 2

def load_patient_data(file_path, node_to_idx_map):
    patient_data = []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            hpo_ids = line.strip().split(',')
            patient_data.append([node_to_idx_map[hpo_id] for hpo_id in hpo_ids])
    return patient_data

def map_idx_to_embeddings(idx_list, embeddings):
    return [embeddings[idx] for idx in idx_list]

def random_similarity_scores(A, embeddings, num_samples=100):
    sim_scores = []
    for _ in range(num_samples):
        idx = np.random.choice(len(embeddings), len(A), replace=False)
        sim_scores.append(sym_set_similarity(A, [embeddings[_] for _ in idx]))
    return sim_scores

def random_similarity_scores_set(S, embeddings, num_samples=100):
    sim_scores = []
    for A in S:
        sim_scores.append(np.mean(random_similarity_scores(map_idx_to_embeddings(A,embeddings), embeddings, num_samples)))
    return sim_scores

def get_k_hop_neighbours(node, k, neighbours):
    if k == 0:
        return [node]
    if node not in neighbours:
        return []
    result = []
    for n in neighbours[node]:
        result.extend(get_k_hop_neighbours(n, k-1, neighbours))
    return result

class GA_HPO():
    def __init__(self, patient_data, neighbors_map, embeddings, max_hops = 2, verbose = False):
        self.patient_data = patient_data
        self.embeddings = embeddings
        self.max_hops = max_hops # number of hops to consider for gene space relative to patient data
        self.empty_gene = -1
        self.hpo_search_space = self.build_hpo_search_space(patient_data, neighbors_map)
        self.verbose = verbose

    def build_hpo_search_space(self, patient_data, neighbours_map):
        gene_space = []
        # get the set of unique terms in the patient data
        patient_terms = list(set([item for sublist in patient_data for item in sublist]))
        # for each patient term, get the k-hop neighbors and add to gene space
        for term in patient_terms:
            k = 1
            gene_space.append(term)
            while k <= self.max_hops:
                gene_space.extend(self.get_k_hop_neighbours(term, k, neighbours_map))
                k += 1
        gene_space = list(set(gene_space))
        gene_space.append(self.empty_gene)
        return gene_space

    def get_k_hop_neighbours(self, node, k, neighbours):
        if k == 0:
            return [node]
        if node not in neighbours:
            return []
        result = []
        for n in neighbours[node]:
            result.extend(self.get_k_hop_neighbours(n, k-1, neighbours))
        return result

    def fitness_func(self, ga_instance, solution, solution_idx):
        ts = map_idx_to_embeddings([_ for _ in solution if _ != self.empty_gene], self.embeddings)
        fitness = 0
        for patient in self.patient_data:
            ps = map_idx_to_embeddings(patient, self.embeddings)
            fitness += sym_set_similarity(ts, ps)
        return fitness/len(self.patient_data)*100
    
    def on_generation(self, ga_instance):
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
       # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       # print("Fitness of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
       # print("Best solution : ", solution)
       # print("Index of the best solution : ", solution_idx)
       # print("=====================================================")


    def run_ga(self, generations = 100, num_parents_mating = 10, sol_per_pop=20, seed=654321, GA_kwargs={}):
        num_genes = max(len(_) for _ in self.patient_data)
        if self.verbose:
            GA_kwargs['on_generation'] = self.on_generation
        ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           fitness_func=self.fitness_func,
                           sol_per_pop=sol_per_pop,
                           gene_type=int,
                           gene_space=self.hpo_search_space,
                           stop_criteria="reach_95",
                           random_seed=seed,
                           **GA_kwargs
                           )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
       # print("Best solution : ", solution)
       # print("Fitness of the best solution : ", solution_fitness)
       # print("Index of the best solution : ", solution_idx)
        return ga_instance
        

# %% solve for cohort of patients
def phenotype_solver(patient_data, neighbours, embeddings, verbose=False,
                     generations=100, num_parents_mating=5, sol_per_pop=10, seed=654321, GA_kwargs='default'):
    if GA_kwargs == 'default':
        GA_kwargs = {'mutation_probability': 0.2,
                    'crossover_probability': 0.5,
                    'crossover_type': 'single_point',
                    'parent_selection_type': 'sss',
                    'mutation_type': 'random',
                    'mutation_by_replacement': True,
                    'parallel_processing':["thread", 10]}
    ga_hpo = GA_HPO(patient_data, neighbours, embeddings, verbose=verbose)
    ga_instance = ga_hpo.run_ga(generations=generations, num_parents_mating=num_parents_mating, sol_per_pop=sol_per_pop, 
                                seed=seed, GA_kwargs=GA_kwargs)
    return ga_hpo, ga_instance

def term_relevance(term, disease_k_hop_neighbours):
    #assumes disease_k_hop_neighbours dict of form {diseaseterm: {k: [neighbours]}}
    # returns a dict of the form {diseaseterm: relevance} where the relevance is the relevance of the term to the disease term
    term_relevance = {}
    for disease_term in disease_k_hop_neighbours.keys():
        tmp = 0
        if term == disease_term:
            tmp = 1
        else:
            for k, neighbours in disease_k_hop_neighbours[disease_term].items():
                if term in neighbours:
                    tmp = (1/(k+1))
        term_relevance[disease_term] = tmp
    return term_relevance

def solution_term_relvances(solution, disease_k_hop_neighbours, threshold=1/3, epsilon=0.001):
    disease_term_count = len(disease_k_hop_neighbours)
    recovered_terms = []
    
    term_relevances = {}
    extra_terms = [term for term in solution]
    for term in solution:
        term_relevances[term] = term_relevance(term, disease_k_hop_neighbours)
    for disease_term in disease_k_hop_neighbours.keys():
        for term, relevance in term_relevances.items():
            if relevance[disease_term] >= threshold-epsilon:
                recovered_terms.append(disease_term)
                if term in extra_terms:
                    extra_terms.remove(term)
                break
    recovered_terms = list(set(recovered_terms))
    recovered_terms = [idx_2_concept[_] for _ in recovered_terms]
    extra_terms = [idx_2_concept[_] for _ in extra_terms]
    fraction_recovered = len(recovered_terms)/disease_term_count
    return recovered_terms, extra_terms, fraction_recovered

# %%
# # TODO this is the preferred solution over creating the iter_gene_dict object below for multiprocessing, but 
# multiprocessing uses pickle to serialize the function and the function is not pickleable
# this could be modified to use dill instead of pickle

# def make_solution_for_gene_func(base_dir, cohort_size, max_k, generations=100, num_parents_mating=5, sol_per_pop=10, seed=SEED,
#                            verbose=False):
#     def f(gene_id_dict_tpl):
#         find_solution_for_gene(gene_id_dict_tpl, base_dir, cohort_size, max_k, generations, num_parents_mating, sol_per_pop, seed, verbose)
#     return f

def find_solution_for_gene(gene_id_dict_tpl, generations=100, num_parents_mating=5, sol_per_pop=10, seed=SEED, verbose=False):
    base_dir, cohort_size, max_k, gene_id, gene_id_dict = gene_id_dict_tpl
    with open(slurm_log, 'a+') as f:
        f.write(f'gene {gene_id}, {base_dir}\n')
    # next two lines are for testing only
    # if gene_id not in ['10', '16']:
    #     return {}, {}
    patient_data_path = os.path.join(dir_data_root, 'processed', 'simulated_patients_gene_pheno', base_dir, f'{gene_id}.csv')
    patient_data = load_patient_data(patient_data_path, concept_2_idx)
    random.seed(SEED)
    patient_data = random.sample(patient_data, cohort_size)
    gene_hpo_idx = gene_id_dict['hpo_idx']
    results = {gene_id:{}}
    errors = {gene_id:False}
    try:
        ga_hpo, ga_instance = phenotype_solver(patient_data, neighbours, embeddings, verbose=verbose, generations=generations, num_parents_mating=num_parents_mating, sol_per_pop=sol_per_pop, seed=seed)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        #compare solution to gene data
        ts = map_idx_to_embeddings([_ for _ in solution if _ != ga_hpo.empty_gene], ga_hpo.embeddings)
        gs = map_idx_to_embeddings(gene_hpo_idx, ga_hpo.embeddings)
        score = sym_set_similarity(ts, gs)
        gene_hpo_ids = [idx_2_concept[_] for _ in gene_hpo_idx]
        solution_hpo_ids = [idx_2_concept[_] for _ in solution if _ != ga_hpo.empty_gene]
        gene_random_sim = random_similarity_scores(gs, embeddings, num_samples=100)
        gene_random_sim = np.mean(gene_random_sim)
        patient_random_sim = np.mean(random_similarity_scores_set(patient_data, embeddings, num_samples=100))
        disease_term_k_hop_neighbours = {}
        for disease_term in gene_hpo_idx:
            disease_term_k_hop_neighbours[disease_term] = {}
            for k in [_+1 for _ in range(max_k)]:
                disease_term_k_hop_neighbours[disease_term][k] = get_k_hop_neighbours(disease_term, k, neighbours)
        recovered_terms, extra_terms, fraction_recovered = solution_term_relvances(solution, disease_term_k_hop_neighbours, threshold = 1/(max_k+1))
        results[gene_id]['recovered_terms'] = recovered_terms
        results[gene_id]['extra_terms'] = extra_terms
        results[gene_id]['fraction_recovered'] = fraction_recovered
        results[gene_id]['gene_id'] = gene_id
        results[gene_id]['gene_terms'] = gene_hpo_ids
        results[gene_id]['gene_term_count'] = len(gene_hpo_ids)
        results[gene_id]['gene_mean_random_similarity'] = gene_random_sim
        results[gene_id]['solution_terms'] = solution_hpo_ids
        results[gene_id]['solution_term_count'] = len(solution_hpo_ids)
        results[gene_id]['gene_solution_similarity'] = score
        results[gene_id]['patient_mean_term_count'] = np.mean([len(_) for _ in patient_data])
        results[gene_id]['patient_solution_mean_similarity'] = solution_fitness/100.0
        results[gene_id]['patient_mean_random_similarity'] = patient_random_sim
        results[gene_id]['patient_gene_mean_similarity'] = np.mean([sym_set_similarity(gs, map_idx_to_embeddings(_, ga_hpo.embeddings)) for _ in patient_data])
    except:
        errors[gene_id] = True
    return results, errors

# print("Definitions generated ...")

if __name__ == '__main__':
# %% run simulations
    sim_dict = {
        #'optimal': {'hop_probability': [1], 'noise_ratio': 0},
        #'imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0},
        #'imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0},
        #'noise_01': {'hop_probability': [1], 'noise_ratio': 0.1},
        'noise_02': {'hop_probability': [1], 'noise_ratio': 0.5},
        'noise_imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.1},
        'noise_imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.1},
        'noise_imprecision_03': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.5},
        'noise_imprecision_04': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.5},
    }

    # sim_dict = {
    #     'optimal': {'hop_probability': [1], 'noise_ratio': 0},
    #     #'imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0},
    #     # 'imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0},
    #     # 'noise_01': {'hop_probability': [1], 'noise_ratio': 0.1},
    #     # 'noise_02': {'hop_probability': [1], 'noise_ratio': 0.5},
    #     #'noise_imprecision_01': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.1},
    #     #'noise_imprecision_02': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.1},
    #     # 'noise_imprecision_03': {'hop_probability': [.2, .7, 1], 'noise_ratio': 0.5},
    #     # 'noise_imprecision_04': {'hop_probability': [.2, .7, .9, 1], 'noise_ratio': 0.5},
    # }

    max_k = 2 # maximum hops for a recovered term to be considered relevant
    cohort_size = 10 # number of individuals sampled per disease

    dir_output = os.path.join(dir_data_root, 'processed', 'simulated_patients_gene_pheno', 'population_phenotyping_sent_embedding',f'cohort_size_{cohort_size}',f'k_thresh_{max_k}')
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    for k,v in sim_dict.items():
        base_dir = f"{k}_hop_prob_{''.join([str(_)+'_' for _ in v['hop_probability']])}noise_ratio_{v['noise_ratio']}"
        dir_root_patient_data = os.path.join(dir_data_root, 'processed', 'simulated_patients_gene_pheno', base_dir)

        results = {'gene_id':[],
            'gene_terms':[],
            'gene_term_count':[],
            'solution_terms':[],
            'solution_term_count':[],
            'gene_solution_similarity':[],
            'gene_mean_random_similarity':[],
            'patient_mean_term_count':[],
            'patient_solution_mean_similarity':[],
            'patient_mean_random_similarity':[],
            'patient_gene_mean_similarity':[],
            'recovered_terms':[],
            'extra_terms':[],
            'fraction_recovered':[]}
        
        errors = []
        
        # spread the work across multiple processes
        with multiprocessing.Pool(20) as pool:
            iter_gene_dict = [(base_dir, cohort_size, max_k, _[0], _[1]) for _ in gene_dict.items()]
            results_errors = pool.map(find_solution_for_gene, iter_gene_dict)
        # collect the results
        for result, error in results_errors:
            for gene_id, gene_results in result.items():
                if 'gene_id' not in gene_results:
                    continue
                results['gene_id'].append(gene_results['gene_id'])
                results['gene_terms'].append(gene_results['gene_terms'])
                results['gene_term_count'].append(gene_results['gene_term_count'])
                results['solution_terms'].append(gene_results['solution_terms'])
                results['solution_term_count'].append(gene_results['solution_term_count'])
                results['gene_solution_similarity'].append(gene_results['gene_solution_similarity'])
                results['gene_mean_random_similarity'].append(gene_results['gene_mean_random_similarity'])
                results['patient_mean_term_count'].append(gene_results['patient_mean_term_count'])
                results['patient_solution_mean_similarity'].append(gene_results['patient_solution_mean_similarity'])
                results['patient_mean_random_similarity'].append(gene_results['patient_mean_random_similarity'])
                results['patient_gene_mean_similarity'].append(gene_results['patient_gene_mean_similarity'])
                results['recovered_terms'].append(gene_results['recovered_terms'])
                results['extra_terms'].append(gene_results['extra_terms'])
                results['fraction_recovered'].append(gene_results['fraction_recovered'])
            for gene_id, error in error.items():
                if error:
                    errors.append(gene_id)

        # save results to file
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(dir_output, f'{base_dir}_model_{embedding_model}.csv'), index=False)
        df_errors = pd.DataFrame({'error_gene_ids':errors})
        df_errors.to_csv(os.path.join(dir_output, f'errors_{base_dir}_model_{embedding_model}.csv'), index=False)

# %%
