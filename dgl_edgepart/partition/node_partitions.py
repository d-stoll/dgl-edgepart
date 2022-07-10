import random
from pathlib import Path

import pandas as pd
import pyarrow
from pyspark import Row
from pyspark.sql import SparkSession


def assign_node_partitions(partition_file: str, use_pyspark: bool = False):
    if not use_pyspark:
        edges = pyarrow.csv.read_csv(partition_file,
                                     read_options=pyarrow.csv.ReadOptions(column_names=["src", "dst", "part_id"]),
                                     parse_options=pyarrow.csv.ParseOptions(delimiter=' ')).to_pandas()

        src_parts = edges[["src", "part_id"]].rename(columns={"src": "nid"})
        dst_parts = edges[["dst", "part_id"]].rename(columns={"dst": "nid"})

        all_parts = pd.concat([src_parts, dst_parts], ignore_index=True)
        node_part = all_parts.groupby("nid").apply(lambda x: x.sample(1)).reset_index(drop=True)

        output_path = Path(partition_file).parent / "node_parts.csv"
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

        output_path = Path(partition_file).parent / "node_parts.csv"
        node_part = node_part.map(lambda r: Row(r[0])).toDF().toPandas()
        node_part.to_csv(output_path, sep=' ', header=False)
        return node_part


