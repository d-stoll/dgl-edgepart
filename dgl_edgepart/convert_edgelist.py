import json
import os
import time

import dgl
import numpy as np
import pyarrow
import torch as th
from dgl import backend as F, NID, EID
from dgl.distributed.partition import _get_inner_node_mask, _get_inner_edge_mask
from dgl.partition import reshuffle_graph
from pyarrow import csv

from dgl_edgepart.partition.node_partitions import assign_node_partitions
from dgl_edgepart.utils.graph_ops import remove_isolated_nodes
from dgl_edgepart.utils.save import save_partition


def edgepart_file_to_dgl(input_file: str, graph_name: str, num_parts: int, part_method: str, output_dir: str,
                         use_spark: bool):
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    node_part = assign_node_partitions(input_file,use_pyspark=use_spark, use_cache=True)
    print("Select node partitions: {:.3f} seconds".format(time.time() - start))

    start = time.time()
    edges = csv.read_csv(input_file, read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst", "part_id"]),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

    u = edges["src"].to_numpy()
    v = edges["dst"].to_numpy()
    g: dgl.DGLGraph = dgl.graph((u, v))
    print("Read graph from input file: {:.3f} seconds".format(time.time() - start))

    g, node_part = reshuffle_graph(g, node_part["part_id"])
    orig_nids = g.ndata['orig_id']
    orig_eids = g.edata['orig_id']

    parts = {}
    for pid in range(num_parts):
        node_ids = (node_part == pid).nonzero(as_tuple=True)[0].tolist()
        part_edge_mask = (edges["part_id"] == pid) | edges["src"].isin(node_ids) | edges["dst"].isin(node_ids)
        subgraph = dgl.edge_subgraph(g, th.tensor(part_edge_mask), store_ids=True)

        remove_isolated_nodes(subgraph)

        epart_assign = edges["part_id"].loc[subgraph.edata['_ID']].to_numpy()
        subgraph.edata['inner_edge'] = th.from_numpy(np.where(epart_assign == pid, 1, 0))
        subgraph.edata['orig_id'] = F.gather_row(orig_eids, subgraph.edata[EID])

        npart_assign = th.index_select(node_part, 0, subgraph.ndata[NID])
        subgraph.ndata['inner_node'] = (npart_assign == pid).int()
        subgraph.ndata['part_id'] = npart_assign
        subgraph.ndata['orig_id'] = F.gather_row(orig_nids, subgraph.ndata[NID])

        parts[pid] = subgraph

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

        part_dir = os.path.join(output_dir, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {
            'node_feats': os.path.relpath(node_feat_file, output_dir),
            'edge_feats': os.path.relpath(edge_feat_file, output_dir),
            'part_graph': os.path.relpath(part_graph_file, output_dir)}

        save_partition(g, part, part_dir)

    with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)