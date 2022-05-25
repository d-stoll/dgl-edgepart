import argparse

from dgl_edgepart.convert_edgelist import convert_edgelist


def build_parser():
    parser = argparse.ArgumentParser(description='Convert edgelist file with partitions into dgl graph partitions')
    parser.add_argument('--input-file', required=True, type=str,
                        help='The file path that contains the edges with partition results. (Format: <dgl_edgepart> <dst> <part_id>)')
    parser.add_argument('--graph-name', required=True, type=str,
                        help='The graph name')
    parser.add_argument('--num-parts', required=True, type=int,
                        help='The number of partitions')
    parser.add_argument('--output', required=True, type=str,
                        help='The output directory of the partitioned results')

    return parser


def cli():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    print(f"Graph name: {args.graph_name}")
    print(f"Num Parts: {args.num_parts}")
    print(f"Output: {args.output}")

    convert_edgelist(args.input_file, args.graph_name, args.num_parts, args.output)
