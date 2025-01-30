import torch
import numpy as np
import matplotlib.pyplot as plt


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

    print(f'Mean of std across embedding dimension from samples : {torch.mean(torch.std(graph_embedding, dim=-1))}')
    print(
        f'Mean of std across embedding dimension from generated samples : {torch.mean(torch.std(boundary_graph_embedding, dim=-1))}')

    graph_embedding = graph_embedding.unsqueeze(1)
    boundary_graph_embedding = boundary_graph_embedding.unsqueeze(2)

    margin = torch.norm(graph_embedding - boundary_graph_embedding, p=2, dim=0).min().item()

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

    # # Compute pairwise distances between all columns in graph_embedding and boundary_graph_embedding
    # dist_matrix = torch.norm(graph_embedding[:, None, :] - boundary_graph_embedding[:, :, None], p=2, dim=0)
    #
    # # Generate points along the line segment between g1 and g2 using broadcasting
    # lambda_values = torch.linspace(0, 1.0, num_points).unsqueeze(0)  # Shape: (1, num_points)
    #
    # h0 = (graph_embedding[:, : , None] * lambda_values).unsqueeze(1).expand(-1, boundary_graph_embedding.shape[-1], -1, -1)
    # h1 = (boundary_graph_embedding[:, : , None] * (1 -lambda_values)).unsqueeze(2)
    #
    # interpolated_points = h0 + h1 # Shape: (embedding_dim, num_points, batch_size)
    #
    # # Reshape the tensor for model scoring (shape: (num_points, batch_size, embedding_dim))
    # interpolated_points = interpolated_points.permute(3,1,2,0)  # Shape: (num_points, batch_size, embedding_dim)
    #
    # # Batch model scoring
    # y_new_batch = model_scoring_function(embeds=interpolated_points)['probs'].T  # Shape: (num_points, batch_size)
    #
    # # Compute thickness for each pair
    # thickness = []
    # for i in range(graph_embedding.shape[1]):
    #     for j in range(boundary_graph_embedding.shape[1]):
    #         dist = dist_matrix[j, i].item()
    #         thickness_value = dist * (gamma > (y_new_batch[c1, i, j]  - y_new_batch[c1, i, j] )).sum().item() / num_points
    #         thickness.append(thickness_value)

    thickness = []
    num_samples = min(graph_embedding.shape[1], boundary_graph_embedding.shape[1])

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

    #Return the average thickness
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
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

    #get only real values
    eigenvalues = eigenvalues.real
    eigenvalues[eigenvalues < 0] = 0 #eigenvalues should not be less than 0, possibly due to numerical imprecision
    normalised_eigenvalues = eigenvalues / eigenvalues.sum()

    #shannon entropy: sum(p_i * log(p_i))
    shannon_entropy = -torch.sum((normalised_eigenvalues * torch.log(normalised_eigenvalues)).nan_to_num()) #nan_to_num(), because in torch inf * 0 = nan

    #complexity is shannon_entropy / log(embedding_dimension)
    return  shannon_entropy / np.log(boundary_graph_embedding.shape[0])


def get_model_boundary_margin(boundary_graphs,
                              dataset_list_pred,
                              model,
                              original_class_idx,
                              adjacent_class_idx,
                              from_best_boundary_graph=False):


    if from_best_boundary_graph:
        boundary_graphs = get_best_boundary_graph(boundary_graphs,
                                                  original_class_idx,
                                                  adjacent_class_idx,
                                                  key='embeds').T
    else:
        boundary_graphs = boundary_graphs['embeds'].T

    return boundary_margin(dataset_list_pred[original_class_idx].model_transform(model, key='embeds').T,
                           boundary_graphs)


def get_model_boundary_thickness(boundary_graphs,
                                 dataset_list_pred,
                                 model,
                                 original_class_idx,
                                 adjacent_class_idx,
                                 from_best_boundary_graph=False):


    if from_best_boundary_graph:
        boundary_graphs = get_best_boundary_graph(boundary_graphs,
                                                  original_class_idx,
                                                  adjacent_class_idx,
                                                  key='embeds').unsqueeze(1)
    else:
        boundary_graphs = boundary_graphs['embeds'].T

    return boundary_thickness(dataset_list_pred[original_class_idx].model_transform(model, key='embeds').T,
                              boundary_graphs,
                              model,
                              original_class_idx,
                              adjacent_class_idx,
                              gamma=0.75,
                              num_points=50)


def get_best_boundary_graph(boundary_graphs,
                            original_class_idx,
                            adjacent_class_idx,
                            key):

    boundary_graph_probs = boundary_graphs['probs']

    best_graph_idx = torch.argmin((boundary_graph_probs[:, original_class_idx] - 0.5).abs()
                                  + (boundary_graph_probs[:, adjacent_class_idx] - 0.5).abs())

    return boundary_graphs[key][best_graph_idx, :]



def sample_valid_boundary_graphs(trainer,
                                 num_samples,
                                 original_class_idx,
                                 adjacent_class_idx):

    cur_samples = 0

    p_min = 0.45
    p_max = 0.55
    cur_tries = 0
    tries_lim = 10000

    boundary_graphs = None

    while cur_samples < num_samples:

        cur_boundary_graphs = trainer.quantatitive(
            sample_size=num_samples,
            use_train_sampling=True,
            return_model_out=True
        )

        cond1 = torch.logical_and(p_min <= cur_boundary_graphs['probs'][:, original_class_idx],
                                  cur_boundary_graphs['probs'][:, original_class_idx] <= p_max)

        cond2 = torch.logical_and(p_min <= cur_boundary_graphs['probs'][:, adjacent_class_idx],
                                  cur_boundary_graphs['probs'][:, adjacent_class_idx] <= p_max)

        valid_idx = torch.logical_and(cond1, cond2)


        if boundary_graphs is None:
            boundary_graphs = {
                'logits': cur_boundary_graphs['logits'][valid_idx, :],
                'probs': cur_boundary_graphs['probs'][valid_idx, :],
                'embeds': cur_boundary_graphs['embeds'][valid_idx, :],
                'embeds_last': cur_boundary_graphs['embeds_last'][valid_idx, :],
            }
        else:
            for key, value in boundary_graphs.items():
                boundary_graphs[key] = torch.cat((value, cur_boundary_graphs[key][valid_idx, :]), dim=0)

        cur_samples += valid_idx.sum().item()

        if cur_tries % 1000 == 0:
            print(f'Sampled {cur_tries}, successful samples: {cur_samples}, remaining: {num_samples}')

        if cur_tries % tries_lim == 0 and cur_samples == 0:
            p_min -= 0.05
            p_max += 0.05
            print(f'Loosening sampling criteria. p_min: {p_min}, p_max: {p_max}')

        cur_tries += num_samples

    for key, value in boundary_graphs.items():
        boundary_graphs[key] = value[:num_samples, :]

    return boundary_graphs


def evaluate_boundary(dataset,
                      trainers,
                      adjacent_class_pairs,
                      model,
                      num_samples):

    num_classes = len(dataset.GRAPH_CLS)
    boundary_margin_mat = np.zeros((num_classes, num_classes))
    boundary_thickness_mat = np.zeros((num_classes, num_classes))
    dataset_list_pred = dataset.split_by_pred(model)
    complexity = {}

    for trainer, class_pair in zip(trainers, adjacent_class_pairs):

        c1, c2 = class_pair
        boundary_graphs = sample_valid_boundary_graphs(trainer, num_samples, c1, c2)

        pca_analysis(
            dataset_list_pred[c1].model_transform(model, key='embeds'),
            dataset_list_pred[c2].model_transform(model, key='embeds'),
            boundary_graphs['embeds'],
            dataset.name.lower(),
            c1,
            c2
        )
        print(f'Calculating boundary complexity for {c1} and {c2}')
        complexity[class_pair] = boundary_complexity(boundary_graphs['embeds_last'].T).item()

        print(f'Calculating boundary margin for {c1} and {c2}')
        margin = get_model_boundary_margin(boundary_graphs,
                                           dataset_list_pred,
                                           model,
                                           original_class_idx=c1,
                                           adjacent_class_idx=c2,
                                           from_best_boundary_graph=False)

        print(f'Calculating boundary thickness for {c1} and {c2}')
        thickness = get_model_boundary_thickness(boundary_graphs,
                                                 dataset_list_pred,
                                                 model,
                                                 original_class_idx=c1,
                                                 adjacent_class_idx=c2,
                                                 from_best_boundary_graph=False)

        boundary_thickness_mat[c1, c2] = thickness
        boundary_margin_mat[c1, c2] = margin

        print(f'Calculating boundary margin for {c2} and {c1}')
        margin = get_model_boundary_margin(boundary_graphs,
                                           dataset_list_pred,
                                           model,
                                           original_class_idx=c2,
                                           adjacent_class_idx=c1,
                                           from_best_boundary_graph=False)

        print(f'Calculating boundary thickness for {c2} and {c1}')
        thickness = get_model_boundary_thickness(boundary_graphs,
                                                 dataset_list_pred,
                                                 model,
                                                 original_class_idx=c2,
                                                 adjacent_class_idx=c1,
                                                 from_best_boundary_graph=False)

        boundary_margin_mat[c2, c1] = margin
        boundary_thickness_mat[c2, c1] = thickness

    return dict(
        boundary_margin=boundary_margin_mat,
        boundary_thickness=boundary_thickness_mat,
        boundary_complexity=complexity
    )


def pca_analysis(embedding_class_1, embedding_class_2, boundary_embedding, dataset_name, c1, c2, k=2, num_samples=100):

    c1_size = embedding_class_1.shape[0] # (num samples * dimension)
    c2_size = embedding_class_2.shape[0]
    b_size = boundary_embedding.shape[0]

    embeddings = torch.cat((embedding_class_1, embedding_class_2, boundary_embedding), dim=0).T
    U, S, V = torch.svd(torch.cov(embeddings))

    principal_components = U[:, :k]
    embedding_pca = torch.mm((embeddings - torch.mean(embeddings, dim=0)).T, principal_components)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))  # Set a larger figure size
    plt.title('PCA Analysis', fontsize=14, fontweight='bold')

    # Class 1
    plt.scatter(
        embedding_pca[:c1_size, 0][torch.randperm(c1_size)[:min(num_samples, c1_size)]],
        embedding_pca[:c1_size, 1][torch.randperm(c1_size)[:min(num_samples, c1_size)]],
        color='royalblue',
        label=f'Class {c1}',
        alpha=0.6,
        s=50  # Marker size
    )

    # Class 2
    plt.scatter(
        embedding_pca[c1_size:c1_size + c2_size, 0][torch.randperm(c2_size)[:min(num_samples, c2_size)]],
        embedding_pca[c1_size:c1_size + c2_size, 1][torch.randperm(c2_size)[:min(num_samples, c2_size)]],
        color='seagreen',
        label=f'Class {c2}',
        alpha=0.6,
        s=50
    )

    # Boundary
    plt.scatter(
        embedding_pca[c1_size + c2_size:, 0][torch.randperm(b_size)[:min(num_samples, b_size)]],
        embedding_pca[c1_size + c2_size:, 1][torch.randperm(b_size)[:min(num_samples, b_size)]],
        color='crimson',
        label='Boundary',
        alpha=1,
        edgecolors='black',  # Add black edges
        linewidths=1.5,  # Thicker edges
        s=50  # Larger marker size for emphasis
    )

    # Axis labels
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)

    # Grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend
    plt.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9)

    # Save the plot
    plt.savefig(f'./figures/{dataset_name}/pca_{c1}_{c2}_embdes.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_sampler_distribution(trainer, dataset_name, class_pair):

    c1, c2 = class_pair
    p = trainer.sampler.p.detach().numpy()
    num_p = p.shape[1]

    fig, axs = plt.subplots(1, num_p + 1, figsize=(15, 6))
    bins = np.linspace(0, 1, 20)

    axs[0].hist(
        trainer.sampler.theta.detach().numpy(),
        bins=bins,
        color='blue',
        edgecolor='black',
        alpha=0.7,
        align='mid'
    )
    axs[0].set_title("Histogram of \nTheta (Edge Weights)", fontsize=12)
    axs[0].set_xlabel("Probability", fontsize=10)
    axs[0].set_ylabel("Density", fontsize=10)
    axs[0].set_xlim(0, 1)

    for i in range(num_p):
        axs[i + 1].hist(
            p[:, i],
            bins=bins,
            color='blue',
            edgecolor='black',
            alpha=0.7,
            align='mid'
        )
        axs[i + 1].set_title(f"Histogram of \np{i} (Feature {i} weights) \nclass pair ({c1},{c2})", fontsize=12)
        axs[i + 1].set_xlabel("Probability", fontsize=10)
        axs[i + 1].set_ylabel("Density", fontsize=10)
        axs[i + 1].set_xlim(0, 1)

    for ax in axs:
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    fig.savefig(f'./figures/{dataset_name}/sampler_hist_{c1}_{c2}.png', dpi=300, bbox_inches='tight')
    plt.show()




