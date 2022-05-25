import importlib

from dgl_edgepart.convert_edgelist import convert_edgelist


def test_convert_edgelist():
    with importlib.resources.path("tests.resources", "test_partitioned.txt") as edgelist_file:
        convert_edgelist(str(edgelist_file), "test", 3, "test")