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
                       c1,
                       c2,
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
    thickness = []

    for idx in range(num_samples):

        g1 = graph_embedding[:, idx]

        if boundary_graph_embedding.shape[1] > 1:
            g2 = boundary_graph_embedding[:, idx]
        else:
            g2 = boundary_graph_embedding[:, 0]

        ## We use l2 norm to measure distance
        dist = torch.norm(g1 - g2, p=2)

        new_batch = []

        ## Sample some points from each segment
        ## This number can be changed to get better precision

        for lmbd in np.linspace(0, 1.0, num=num_points):
            new_batch.append(g1 * lmbd + g2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)

        y_new_batch = model_scoring_function(embeds=new_batch)['probs'].T

        thickness.append(dist.item() * (gamma > y_new_batch[c1, :] - y_new_batch[c2, :]).sum().item() / num_points)

    return np.mean(thickness)


def boundary_complexity(boundary_graph_embedding):
    
    """
    Args:
        boundary_graph_embedding (torch.Tensor): A tensor of shape (embedding_dimension, num_boundary_graphs)
                                                 containing the embeddings of the boundary graphs

    Returns:
        complexity (float): The complexity of the decision boundary, a value between 0 and 1.
    """

    #increase floating point precision
    boundary_graph_embedding = boundary_graph_embedding.to(dtype=torch.float64)

    #eigenvalue obtained from PCA decomposition according to paper
    covariance_matrix = torch.cov(boundary_graph_embedding)
    eigenvalues, _ = torch.linalg.eig(covariance_matrix)

    #get only real values
    eigenvalues = eigenvalues.real
    normalised_eigenvalues = eigenvalues / eigenvalues.sum()

    #shannon entropy: sum(p_i * log(p_i))
    normalised_eigenvalues[normalised_eigenvalues < 0] = 0 #eigenvalues should not be less than 0, possibly due to numerical imprecision
    shannon_entropy = -torch.sum((normalised_eigenvalues * torch.log(normalised_eigenvalues)).nan_to_num()) #nan_to_num(), because in torch inf * 0 = nan

    #complexity is shannon_entropy / log(embedding_dimension)
    return  shannon_entropy / np.log(boundary_graph_embedding.shape[0])


def get_model_boundary_margin(trainer,
                              dataset_list_pred,
                              model,
                              original_class_idx,
                              adjacent_class_idx,
                              num_samples,
                              from_best_boundary_graph=False):


    boundary_graphs = sample_valid_boundary_graphs(trainer, num_samples, original_class_idx, adjacent_class_idx)

    if from_best_boundary_graph:
        boundary_graphs = get_best_boundary_graph(trainer,
                                                   boundary_graphs,
                                                   original_class_idx,
                                                   adjacent_class_idx, 'embeds').T
    else:
        boundary_graphs = trainer.predict_batch(boundary_graphs)['embeds'].T

    return boundary_margin(dataset_list_pred[original_class_idx].model_transform(model, key='embeds').T,
                           boundary_graphs)


def get_model_boundary_thickness(trainer,
                                 dataset_list_pred,
                                 model,
                                 original_class_idx,
                                 adjacent_class_idx,
                                 num_samples,
                                 from_best_boundary_graph=False):

    boundary_graphs = sample_valid_boundary_graphs(trainer, num_samples, original_class_idx, adjacent_class_idx)

    if from_best_boundary_graph:
        boundary_graphs = get_best_boundary_graph(trainer,
                                                  boundary_graphs,
                                                  original_class_idx,
                                                  adjacent_class_idx, 'embeds').unsqueeze(1)
    else:
        boundary_graphs = trainer.predict_batch(boundary_graphs)['embeds'].T

    return boundary_thickness(dataset_list_pred[original_class_idx].model_transform(model, key='embeds').T,
                              boundary_graphs,
                              model,
                              original_class_idx,
                              adjacent_class_idx,
                              gamma=0.75,
                              num_points=50)


def get_best_boundary_graph(trainer,
                            boundary_graphs,
                            original_class_idx,
                            adjacent_class_idx,
                            key):

    boundary_predicted_batch = trainer.predict_batch(boundary_graphs)
    boundary_graph_probs = boundary_predicted_batch['probs']

    best_graph_idx = torch.argmin((boundary_graph_probs[:, original_class_idx] - 0.5).abs()
                                  + (boundary_graph_probs[:, adjacent_class_idx] - 0.5).abs())

    return boundary_predicted_batch[key][best_graph_idx, :]


def get_model_complexity(trainer,
                         original_class_idx,
                         adjacent_class_idx,
                         num_samples):


    boundary_graphs = sample_valid_boundary_graphs(trainer, num_samples, original_class_idx, adjacent_class_idx)
    boundary_graphs = trainer.predict_batch(boundary_graphs)['embeds_last'].T

    return boundary_complexity(boundary_graphs)


def sample_valid_boundary_graphs(trainer,
                                 num_samples,
                                 original_class_idx,
                                 adjacent_class_idx):

    boundary_graphs = []
    cur_samples = 0

    p_min = 0.45
    p_max = 0.55

    while cur_samples < num_samples:
        cur_boundary_graph = trainer.evaluate(bernoulli=True)
        probs = trainer.predict(cur_boundary_graph)['probs']

        if p_min <= probs[0][original_class_idx] <= p_max and p_min <= probs[0][adjacent_class_idx] <= p_max:
            boundary_graphs.append(cur_boundary_graph)
            cur_samples += 1

    return boundary_graphs