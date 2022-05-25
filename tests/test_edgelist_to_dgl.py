import importlib.resources

from dgl_edgepart.tools.edgelist_to_dgl import edgelist_to_dgl


def test_edgelist_to_dgl():
    with importlib.resources.path("tests.resources", "test.txt") as edgelist_file:
        g = edgelist_to_dgl(str(edgelist_file))

        print(g)
        print(g.ndata['label'])
        print(g.adj())
