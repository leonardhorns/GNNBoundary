import networkx as nx
import pandas as pd
import torch_geometric as pyg

from ..gnn_xai_common.datasets import BaseGraphDataset
from ..gnn_xai_common.datasets.utils import default_ax, unpack_G


class MultiRedditDataset(BaseGraphDataset):
    NODE_CLS = {
        0: 'There is only one class'
    }

    GRAPH_CLS = {
        0: 'Class 1',
        1: 'Class 2',
        2: 'Class 3',
        3: 'Class 4',
        4: 'Class 5',
        5: 'Class 6',
        6: 'Class 7',
        7: 'Class 8',
        8: 'Class 9',
        9: 'Class 10',
        10: 'Class 11',
    }

    def __init__(self, *,
                 name='REDDIT-MULTI-12K',
                 url='https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-MULTI-12K.zip',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["REDDIT-MULTI-12K/REDDIT-MULTI-12K_A.txt",
                "REDDIT-MULTI-12K/REDDIT-MULTI-12K_graph_indicator.txt",
                "REDDIT-MULTI-12K/REDDIT-MULTI-12K_graph_labels.txt"]

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_zip(f'{self.raw_dir}/REDDIT-MULTI-12K.zip', self.raw_dir)

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) > 0

        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, {i: 0 for i in range(len(graph_idx))},
                               name='label')  # Need to set all node labels to 0, do so using dictionary comprehension
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        return unpack_G(super_G)

    @default_ax
    def draw(self, G, pos=None, label=False, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=300,
                               edgecolors='black')

        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=1, edge_color='tab:gray')

    def process(self):
        super().process()