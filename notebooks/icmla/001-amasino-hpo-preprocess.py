# %% imports
import json
import csv

# %% file paths
file_hpo = '../../data/external/hpo/hp.json'
file_nodes = '../../data/processed/hpo_class_nodes.csv'
file_edges = '../../data/processed/hpo_is_a_edges.csv'

# %% read hpo file
with open(file_hpo, 'r') as f:
    hpo = json.load(f)

# %% extract edges for is_a relationships
edges = hpo['graphs'][0]['edges']
node_map = {}
is_a_edges = []
node_count = 0
for edge in edges:
    if edge['pred'] == 'is_a':
        source = edge['sub'].split('/')[-1]
        dest = edge['obj'].split('/')[-1]
        # index the source nodes first to ensure that the node indices are contiguous
        if source not in node_map:
            node_map[source] = node_count
            node_count += 1
        if dest not in node_map:
            node_map[dest] = node_count
            node_count += 1
        is_a_edges.append((node_map[source], node_map[dest]))

# %% write is_a edges to file
with open(file_edges, 'w+') as out_file:
    out_file.write('source_id,destination_id\n')
    for tpl in is_a_edges:
        out_file.write(f"{tpl[0]},{tpl[1]}\n")

# %%
nodes = hpo['graphs'][0]['nodes']
node_data = []
for node in nodes:
    if 'type' in node and node['type'] == 'CLASS':
        node_id = node['id'].split('/')[-1]
        if node_id in node_map:
            node_idx = node_map[node_id]
            node_label = node['lbl']
            if 'meta' in node and 'definition' in node['meta']:
                node_def = node['meta']['definition']['val']
            else:
                node_def = ''
            node_data.append((node_idx, node_id, node_label, node_def))
node_data = sorted(node_data, key=lambda x: x[0])
# write nodes to file
with open(file_nodes, 'w+') as out_file:
    out_file.write('node_idx,hpo_id,label,definition\n')
    for tpl in node_data:
        out_file.write(f'{tpl[0]},{tpl[1]},"{tpl[2]}","{tpl[3]}"\n')
# %%
