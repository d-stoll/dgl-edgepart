import dgl


def test_dgl_edgepart_cli():
    pass

def test_dgl_partitions():
    (g,), _ = dgl.load_graphs("test/part0/graph.dgl")
    print(g)

    (g,), _ = dgl.load_graphs("test/part1/graph.dgl")
    print(g)