import importlib

from dgl_edgepart.convert_edgelist import edgepart_file_to_dgl


def test_convert_edgelist():
    with importlib.resources.path("tests.resources", "test_partitioned.txt") as edgelist_file:
        edgepart_file_to_dgl(str(edgelist_file), "test", num_parts=3, part_method="custom", output_dir="test")