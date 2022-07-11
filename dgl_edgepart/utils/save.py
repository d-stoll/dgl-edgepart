import os

import dgl
from dgl.data import save_tensors
from dgl.distributed.partition import _get_inner_edge_mask, _get_inner_node_mask
from dgl import backend as F, NID, EID, save_graphs


def save_partition(g: dgl.DGLGraph, part: dgl.DGLGraph, part_dir: str):
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

    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    save_tensors(node_feat_file, node_feats)
    save_tensors(edge_feat_file, edge_feats)
    save_graphs(part_graph_file, [part])