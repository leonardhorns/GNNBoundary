import torch
import numpy as np

def boundary_margin(graph_embedding,
                    boundary_graph_embedding):

    """
    Args:
        graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling layer
        boundary_graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling
                                                 layer of boundary graph

    Returns:
        margin (float) : class boundary margin
    """

    # shuffle embeddings around to generate random ordering
    num_samples = min(graph_embedding.shape[1], boundary_graph_embedding.shape[1])

    graph_embedding = graph_embedding[:, torch.randperm(graph_embedding.shape[1])]
    boundary_graph_embedding = boundary_graph_embedding[:, torch.randperm(boundary_graph_embedding.shape[1])]

    margin = float('inf')

    #iterate through pairs of graph embeddings
    for idx in range(num_samples):
        g1 = graph_embedding[:, idx]
        g2 = boundary_graph_embedding[:, idx]

        #calculate margin
        cur_margin = torch.norm(g2 - g1, p=2).item()

        #update margin
        if cur_margin < margin:
            margin = cur_margin

    return margin


def boundary_thickness(graph_embedding,
                       boundary_graph_embedding,
                       model_scoring_function,
                       class_pair_idx,
                       gamma=0.75,
                       num_points=50):

    """
    Args:
        graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling layer
        boundary_graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling
                                                 layer of boundary graph
        class_pair_idx (tuple(int)) : tuple of class pair idx
        model_scoring_function () : MLP layer after embedding layer
        gamma (float) : hyperparameter
        num_points (int) : number of points used for interpolation

    Returns:
        thickness (float) : boundary thickness margin
    """

    #shuffle data around
    num_samples = min(graph_embedding.shape[1], boundary_graph_embedding.shape[1])

    graph_embedding = graph_embedding[:, torch.randperm(graph_embedding.shape[1])]
    boundary_graph_embedding = boundary_graph_embedding[:, torch.randperm(boundary_graph_embedding.shape[1])]

    for idx in range(num_samples):

        g1 = graph_embedding[:, idx]
        g2 = boundary_graph_embedding[:, idx]

        ## We use l2 norm to measure distance
        dist = torch.norm(g1 - g2, p=2)

        new_batch = []

        ## Sample some points from each segment
        ## This number can be changed to get better precision

        for lmbd in np.linspace(0, 1.0, num=num_points):
            new_batch.append(g1 * lmbd + g2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)

        y_new_batch = model_scoring_function(new_batch)

        c1, c2 = class_pair_idx

        #assuming that y_new_batch is off dimension (num_classes * num_samples)
        boundary_thickness = np.where(gamma > y_new_batch[c1, :] - y_new_batch[c2, :])

        return dist.item() * np.sum(boundary_thickness) / num_points


def boundary_complexity(boundary_graph_embedding):
    
    """
    Args:
        boundary_graph_embedding (torch.Tensor): A tensor of shape (embedding_dimension, num_boundary_graphs)
                                                 containing the embeddings of the boundary graphs
                                                 TODO the paper mentions taking phi_l-1, which suggests that the final pooling layer is skipped. Why exactly?

    Returns:
        complexity (float): The complexity of the decision boundary, a value between 0 and 1.
    """
    covariance_matrix = torch.cov(boundary_graph_embedding.T)
    eigenvalues, _ = torch.linalg.eigh(covariance_matrix)

    assert torch.all(eigenvalues > 0) # Numerical imprecision might cause negative eigenvalues
    normalised_eigenvalues = eigenvalues / eigenvalues.sum()

    numerator = -torch.sum(normalised_eigenvalues * torch.log(normalised_eigenvalues))

    embedding_dimension = boundary_graph_embedding.shape[1]
    complexity = (numerator / torch.log(torch.tensor(embedding_dimension, dtype=torch.float32))).item()

    return complexity
