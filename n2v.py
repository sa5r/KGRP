"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import networkx as nx
import argparse
from node2vec import Node2Vec


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required = True, )
    parser.add_argument('--dimensions', required = True, type = int )
    parser.add_argument('--walk_length', required = True, type = int )
    parser.add_argument('--walks', required = True, type = int )
    parser.add_argument('--window', required = True, type = int )
    args = parser.parse_args()


    G = nx.MultiGraph()
    with open( args.data_path + '/train.tsv', 'r') as f:
        for line in f:
            items = line.split('\t')
            if len(items) > 3:
                print('error', items)
                break
            head = items[0].strip()
            tail = items[2].strip()
            rel = items[1].strip()
            G.add_node(head)
            G.add_node(tail)
            G.add_edge(head, tail, )
    print(G)

    node2vec = Node2Vec(G, dimensions=args.dimensions, walk_length=args.walk_length,
                        num_walks=args.walks, workers=4)
    model = node2vec.fit(window=args.window, min_count=1)
    model.wv.save_word2vec_format('n2v_embeddings.csv')

if __name__ == "__main__": main()