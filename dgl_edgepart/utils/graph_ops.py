import dgl


def remove_isolated_nodes(g: dgl.DGLGraph):
    isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
    g.remove_nodes(isolated_nodes)
