import dgl
import pyarrow
from pyarrow import csv
import torch as th


def edgelist_to_dgl(path: str, feat_size: int = 128, nlabels: int = 10, train_split: float = 0.6,
                    val_split: float = 0.2):
    """Converts an edgelist file to a DGL graph with random feature vectors and train/val/test split"""
    assert train_split + val_split <= 1

    edges = csv.read_csv(path, read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst"]),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

    u = th.tensor(edges["src"])
    v = th.tensor(edges["dst"])
    g = dgl.graph((u, v))
    n = g.number_of_nodes()

    node_features = th.randn([n, feat_size], dtype=th.float32)
    train_mask = th.zeros(g.number_of_nodes(), dtype=th.bool)
    val_mask = th.zeros(g.number_of_nodes(), dtype=th.bool)
    test_mask = th.zeros(g.number_of_nodes(), dtype=th.bool)
    labels = th.randint(nlabels, [n], dtype=th.int64)

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    g.ndata['feat'] = node_features
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    g.ndata['label'] = labels

    return g



