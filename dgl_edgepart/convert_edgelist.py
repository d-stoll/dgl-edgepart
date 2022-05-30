import json
import os

import dgl
import numpy as np
import pandas as pd
import pyarrow
import torch as th
from dgl import backend as F, NID, EID, save_graphs
from dgl import partition_graph_with_halo
from dgl.data import save_tensors
from dgl.distributed.partition import _get_inner_node_mask, _get_inner_edge_mask
from pyarrow import csv


def edgepart_file_to_dgl(input_file: str, graph_name: str, num_parts: int, part_method: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    edges = csv.read_csv(input_file, read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst", "part_id"]),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

    u = edges["src"].to_numpy()
    v = edges["dst"].to_numpy()
    g: dgl.DGLGraph = dgl.graph((u, v))

    src_parts = edges[["src", "part_id"]].rename(columns={"src": "nid"})
    dst_parts = edges[["dst", "part_id"]].rename(columns={"dst": "nid"})

    # All partitions a node can be assigned
    node_all_parts = pd.concat([src_parts, dst_parts], ignore_index=True)
    # One partition for each node
    node_part = node_all_parts.groupby("nid").apply(lambda x: x.sample(1)).reset_index(drop=True)
    node_part_tensor = th.tensor(node_part["part_id"].values)

    parts, orig_nids, orig_eids = partition_graph_with_halo(g, node_part_tensor, 1, reshuffle=True)

    node_map_val = {}
    edge_map_val = {}
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        val = []
        node_map_val[ntype] = []
        for i in parts:
            inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
            val.append(F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0)))
            inner_nids = F.boolean_mask(parts[i].ndata[NID], inner_node_mask)
            node_map_val[ntype].append([int(F.as_scalar(inner_nids[0])), int(F.as_scalar(inner_nids[-1])) + 1])
        val = np.cumsum(val).tolist()
        assert val[-1] == g.number_of_nodes(ntype)
    for etype in g.etypes:
        etype_id = g.get_etype_id(etype)
        val = []
        edge_map_val[etype] = []
        for i in parts:
            inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
            val.append(F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0)))
            inner_eids = np.sort(F.asnumpy(F.boolean_mask(parts[i].edata[EID], inner_edge_mask)))
            edge_map_val[etype].append([
                int(inner_eids[0]),
                int(inner_eids[-1]) + 1])
        val = np.cumsum(val).tolist()
        assert val[-1] == g.number_of_edges(etype)

    for ntype in node_map_val:
        val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
        assert np.all(val[:-1] <= val[1:])
    for etype in edge_map_val:
        val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
        assert np.all(val[:-1] <= val[1:])

    ntypes = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype: g.get_etype_id(etype) for etype in g.etypes}

    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': 1,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val,
                     'ntypes': ntypes,
                     'etypes': etypes}

    for part_id in range(num_parts):
        part = parts[part_id]

        node_feats = {}
        edge_feats = {}

        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)

            ndata_name = 'orig_id'
            inner_node_mask = _get_inner_node_mask(part, ntype_id)
            local_nodes = F.boolean_mask(part.ndata[ndata_name], inner_node_mask)

            for name in g.nodes[ntype].data:
                if name in [NID, 'inner_node']:
                    continue
                node_feats[ntype + '/' + name] = F.gather_row(g.nodes[ntype].data[name],
                                                              local_nodes)

        for etype in g.etypes:
            etype_id = g.get_etype_id(etype)
            edata_name = 'orig_id'
            inner_edge_mask = _get_inner_edge_mask(part, etype_id)
            local_edges = F.boolean_mask(part.edata[edata_name], inner_edge_mask)

            for name in g.edges[etype].data:
                if name in [EID, 'inner_edge']:
                    continue
                edge_feats[etype + '/' + name] = F.gather_row(g.edges[etype].data[name],
                                                              local_edges)
        part_dir = os.path.join(output_dir, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {
            'node_feats': os.path.relpath(node_feat_file, output_dir),
            'edge_feats': os.path.relpath(edge_feat_file, output_dir),
            'part_graph': os.path.relpath(part_graph_file, output_dir)}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        save_graphs(part_graph_file, [part])

    with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)