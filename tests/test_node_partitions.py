import importlib.resources
import pathlib

from dgl_edgepart.partition.node_partitions import assign_node_partitions


def test_assign_node_partitions():
    with importlib.resources.path("tests.resources", "test_partitioned.txt") as edgelist_file:
        assign_node_partitions(str(edgelist_file), use_pyspark=True)