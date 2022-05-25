import os
import json
import time
import argparse
import numpy as np
import dgl
import torch as th
import pyarrow
import pandas as pd
from dgl.sparse import libra2dgl_build_adjlist
from pyarrow import csv


def convert_edgelist(input_file: str, graph_name: str, num_parts: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    num_edges = 0
    num_nodes = 0

    edges = csv.read_csv(input_file, read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst", "part_id"]),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()
    tmp_dir = output_dir + '/' + graph_name + '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    for _, (part_id, part_edges) in enumerate(edges.groupby("part_id")):
        part_edges[["src", "dst"]].to_csv(f"{tmp_dir}/part-{part_id}.tmp", index=False)

    for part_id in range(num_parts):
        part_dir = output_dir + '/part' + str(part_id)
        os.makedirs(part_dir, exist_ok=True)

        part_edges = csv.read_csv(f"{tmp_dir}/part-{part_id}.tmp")

        src_ids, dst_ids = part_edges.columns[0].to_numpy(), part_edges.columns[1].to_numpy()
        assert len(src_ids) == len(dst_ids)
        num_nodes += len(src_ids)

        print('There are {} edges in partition {}'.format(len(src_ids), part_id))

        nids = np.concatenate([src_ids, dst_ids])
        uniq_ids, idx, inverse_idx = np.unique(nids, return_index=True, return_inverse=True)

        local_src_id, local_dst_id = np.split(inverse_idx[:len(src_ids) * 2], 2)
        compact_g = dgl.graph((local_src_id, local_dst_id))

        num_nodes_part = compact_g.number_of_nodes()

        print('|V|={}'.format(num_nodes_part))
        print('|E|={}'.format(compact_g.number_of_edges()))



        dgl.save_graphs(part_dir + '/graph.dgl', [compact_g])

    part_metadata = {
        'graph_name': graph_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'part_method': 'custom',
        'num_parts': num_parts,
        'halo_hops': 1
    }

    for part_id in range(num_parts):
        part_dir = 'part' + str(part_id)
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {
            'node_feats': node_feat_file,
            'edge_feats': edge_feat_file,
            'part_graph': part_graph_file}
    with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)