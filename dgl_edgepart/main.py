import argparse

from dgl_edgepart.convert_edgelist import edgepart_file_to_dgl


def build_parser():
    parser = argparse.ArgumentParser(description='Convert edgelist file with partitions into dgl graph partitions')
    parser.add_argument('-i', '--input-file', required=True, type=str,
                        help='The file path that contains the edges with partition results. (Line format: <src> <dst> '
                             '<part_id>)')
    parser.add_argument('-g', '--graph-name', required=True, type=str,
                        help='The name of the graph to be used by later in DGL scripts.')
    parser.add_argument('-m', '--part-method', required=True, type=str,
                        help='The partition method used for creating the input file.')
    parser.add_argument('-p', '--num-parts', required=True, type=int,
                        help='The number of partitions.')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='The output directory of the partitioned results.')
    parser.add_argument('--use-spark', required=False, type=bool,
                        help='Use PySpark to parallelize Pandas computations.')

    return parser


def cli():
    parser = build_parser()
    args = parser.parse_args()

    print(f"=================== Input parameters ===================")
    print(f"Input file: {args.input_file}")
    print(f"Graph name: {args.graph_name}")
    print(f"Part method: {args.part_method}")
    print(f"Num Parts: {args.num_parts}")
    print(f"Output directory: {args.output}")
    print(f"Use PySpark: {args.use_spark}")
    print(f"========================================================")

    edgepart_file_to_dgl(args.input_file, args.graph_name, args.num_parts, args.part_method, args.output,
                         args.use_spark)
