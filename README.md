<h1 align="center">
  DGL Edge-Partitioning Tool
</h1>

<h4 align="center">A CLI tool to convert edge-partitioned graphs into DistDGL compatible files.</h4>

## Usage

The input file has to an edgelist file with partitions as third column. The edges have to be sorted by partition in 
ascending order. Partitions have to start at Example:

```
0 1 0
0 2 0
1 2 1
2 3 1
...
```

If the file is not sorted by the partitions, you can use this simple bash script:

```shell
sort -k 3 input.txt > input-sorted.txt
```

After making sure the input file is in the right format, you can use the ``dgl-edgepart` command to create the DGL files:

```shell
dgl-edgepart [-h] -i INPUT_FILE -g GRAPH_NAME -m PART_METHOD -p NUM_PARTS -o OUTPUT [--use-spark]

Convert edgelist file with partitions into dgl graph partitions

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        The file path that contains the edges with partition results. (Line format: <src> <dst>
                        <part_id>)
  -g GRAPH_NAME, --graph-name GRAPH_NAME
                        The name of the graph to be used by later in DGL scripts.
  -m PART_METHOD, --part-method PART_METHOD
                        The partition method used for creating the input file.
  -p NUM_PARTS, --num-parts NUM_PARTS
                        The number of partitions.
  -o OUTPUT, --output OUTPUT
                        The output directory of the partitioned results.
  --use-spark           Use PySpark to parallelize Pandas computations.
```

Example:

```shell
dgl-edgepart -i input-sorted.txt -g example-graph -m hdrf -p 8 -o example-graph --use-spark
```

## Installation

### Requirements

* Python 3.10+
* Poetry

Make sure Poetry is installed on your system: https://python-poetry.org/docs/#installation

You can install Poetry on Ubuntu (or other Linux Distros) with just a single bash command:

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

This tool requires Python 3.10+. For installing 3.10 on Ubuntu 20.04 I recommend using pyenv: https://github.com/pyenv/pyenv

### Build

After installing Poetry / Python 3.10, the project dependencies and virtualenv can be created by executing:

```shell
poetry install
```

To make the `dgl-edgpart` command available in your console, activate the virtualenv:

```{bash}
poetry shell
```

