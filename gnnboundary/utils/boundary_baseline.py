import random
import numpy as np
import networkx as nx


class BaselineGenerator:

    def __init__(self, dataset, class_pairs, seed=12345):
        """

        :param dataset: list of dataset split by predicted class
        :param class_pairs: list of class idx pair
        :param seed: seed
        """
        assert len(class_pairs) == 2

        random.seed(seed)
        np.random.seed(42)

        self.dataset_1 = dataset[class_pairs[0]]
        self.dataset_2 = dataset[class_pairs[1]]


    def sample(self, sample_size):
        """

        :param sample_size: int of sample size to be generated
        :return: samples: list of nx Graphs
        """

        samples = []

        rand_idx_1 = np.random.randint(len(self.dataset_1), size=sample_size)
        rand_idx_2 = np.random.randint(len(self.dataset_2), size=sample_size)

        for idx_1, idx_2 in zip(rand_idx_1, rand_idx_2):
            G1 = self.convert_to_nx_graph(self.dataset_1[idx_1])
            edges_count = np.max(np.array(self.dataset_1[idx_1].edge_index)) + 1
            G2 = self.convert_to_nx_graph(self.dataset_2[idx_2], edges_count)

            #combine G1 and G2 together
            G = nx.compose(G1, G2)

            #choose 2 random node from each graph
            node_1 = random.choice(list(G1.nodes))
            node_2 = random.choice(list(G2.nodes))

            #join by connecting edge
            G.add_edge(node_1, node_2)

            samples.append(G)

        return samples

    @staticmethod
    def convert_to_nx_graph(dataset_graph, start_index=0):

        #initialise graph
        G = nx.Graph()

        #add edges
        G.add_edges_from([(int(dataset_graph.edge_index[0, idx]) + start_index, int(dataset_graph.edge_index[1, idx]) + start_index) for idx in range(dataset_graph.num_edges)])

        #add features
        for i, features in enumerate(dataset_graph.x):
            G.nodes[i + start_index]['features'] = features
            G.nodes[i + start_index]['label'] = int(np.argmax(features))
        return G

