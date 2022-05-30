import importlib

import dgl.distributed

from dgl_edgepart.tools.edgelist_to_dgl import edgelist_to_dgl


def test_dgl_partitioning():
    with importlib.resources.path("tests.resources", "test.txt") as edgelist_file:
        g = edgelist_to_dgl(str(edgelist_file))
        dgl.distributed.partition_graph(g, graph_name="test-graph", num_parts=3, out_path="3part_data", part_method="random")
