import os
import random
from pathlib import Path

import pandas as pd
import pyarrow
from pyspark import Row
from pyspark.sql import SparkSession


def assign_node_partitions(partition_file: str, use_pyspark: bool = False, use_cache: bool = True):
    """
    Computes node partitions from a precomputed edge partition file. Each node is assigned randomly to one of
    the partitions of its incident edges.

    :param partition_file: Path to edgelist file with partitions.
    :param use_pyspark: Use PySpark to utilize all CPU cores (or even multiple machines) to compute node partitions.
    :param use_cache: If there is already a node_parts file, just return the content as Pandas DF.
    :return: Pandas Dataframe of shape [n_nodes, 1] with partition assignments for each node.
    """
    output_path = Path(partition_file).parent / "node_parts.csv"

    if Path(output_path).exists():
        if use_cache:
            return pyarrow.csv.read_csv(output_path,
                                        read_options=pyarrow.csv.ReadOptions(column_names=["nid", "part_id"]),
                                        parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()
        else:
            os.remove(output_path)

    if not use_pyspark:
        edges = pyarrow.csv.read_csv(partition_file,
                                     read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst", "part_id"]),
                                     parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

        src_parts = edges[["src", "part_id"]].rename(columns={"src": "nid"})
        dst_parts = edges[["dst", "part_id"]].rename(columns={"dst": "nid"})

        all_parts = pd.concat([src_parts, dst_parts], ignore_index=True)
        node_part = all_parts.groupby("nid").apply(lambda x: x.sample(1)).reset_index(drop=True)

        node_part.to_csv(output_path, sep=' ', header=False)
        return node_part
    else:
        spark = SparkSession.builder.appName("Assign Node Partitions").getOrCreate()
        edges = spark.read.options(delimiter=" ", header=False).csv(partition_file).toDF("src", "dst", "part_id")

        src_parts = edges.select("src", "part_id").withColumnRenamed("src", "nid")
        dst_parts = edges.select("dst", "part_id").withColumnRenamed("dst", "nid")

        all_parts = src_parts.unionAll(dst_parts)

        def sample(iter):
            rs = random.Random()
            return rs.sample(list(iter), 1)

        node_part = all_parts.rdd.map(lambda row: (row.nid, row.part_id)).groupByKey().flatMap(lambda g: sample(g[1]))

        node_part = node_part.map(lambda r: Row(r[0])).toDF().toPandas()
        node_part.columns=["part_id"]
        node_part.to_csv(output_path, sep=' ', header=False)
        return node_part
