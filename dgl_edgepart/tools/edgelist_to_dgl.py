import dgl
import pyarrow
from pyarrow import csv
import torch as th


def edgelist_to_dgl(path: str):
    """Converts an edgelist file to a DGL graph"""
    edges = csv.read_csv(path, read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst"]),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

    u = th.tensor(edges["src"])
    v = th.tensor(edges["dst"])
    g = dgl.graph((u, v))

    return g



