import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.targeted_attack import FGA
import networkx as nx
import numpy as np
import scipy.sparse as sp

def attack(dataset,
           model,
           method,
           sample_idx,
           draw=False):


    attack_method = {
        'meta': Metattack,
    }

    #get adjacent matrix, features, and labels from dataset
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    device = torch.device("cpu")

    try:
        dataset.to(device)
    except ValueError:
        pass

    sampled_data = dataset[sample_idx]

    adj = pyg_utils.to_dense_adj(sampled_data.edge_index)[0]
    features = sampled_data.x
    labels = model(sampled_data)['probs'].argmax(dim=-1)

    model.nclass = len(dataset.GRAPH_CLS)
    model.nfeat = len(dataset.NODE_CLS)
    model.hidden_sizes = [model.conv.hidden_channels, model.lin.in_channels, model.lin.out_channels]
    model.with_relu = True
    model.output = F.one_hot(model(sampled_data)['probs'].argmax(dim=-1), num_classes=len(dataset.GRAPH_CLS))

    try:
        model.to(device)
    except ValueError:
        pass


    attack_model = attack_method[method](
        model=model,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=1,
        train_iters=300,
        lr=0.1,
        momentum=0.9
    )

    attack_model.to(device)

    perturbations = 1 #int(0.05 * (adj.sum() // 2))

    attack_model.attack(
        features,
        adj,
        labels,
        [0],
        [0],
        perturbations,
        ll_constraint=False
    )

    results_dict = dict(
        ground_truth= sampled_data.y.item(),
        before_attack = {},
        after_attack = {},
    )

    model.eval()
    with torch.no_grad():
        original_probs = model(sampled_data)['probs']
        results_dict['before_attack']['class_probs'] = original_probs.tolist()
        results_dict['before_attack']['predicted_class'] =  F.softmax(original_probs).argmax(dim=1).item()

        if draw:
            dataset.draw(convert_to_nx(sampled_data))

        modified_edge_idx, _ = pyg_utils.dense_to_sparse(attack_model.modified_adj)
        sampled_data.edge_index = modified_edge_idx
        attack_probs = model(sampled_data)['probs']

        results_dict['after_attack']['class_probs'] = attack_probs.tolist()
        results_dict['after_attack']['predicted_class'] = F.softmax(attack_probs).argmax(dim=1).item()

        if draw:
            dataset.draw(convert_to_nx(sampled_data))
    return results_dict


def convert_to_nx(graph):
    G = nx.Graph()

    G.add_edges_from(
        [(int(graph.edge_index[0, idx]), int(graph.edge_index[1, idx])) for
         idx in range(graph.num_edges)])

    # add features
    for i, features in enumerate(graph.x):
        G.nodes[i]['features'] = features
        G.nodes[i]['label'] = int(np.argmax(features))
    return G


def adversarial_attack(data, model, criterion, budget=5):
    model.eval()
    perturbed_data = data.clone()

    edge_index_original = perturbed_data.edge_index.clone()

    for _ in range(budget):
        A = torch.sparse_coo_tensor(perturbed_data.edge_index,
                                   torch.ones(perturbed_data.edge_index.size(1)),
                                   (data.num_nodes, data.num_nodes)).to_dense()
        A.requires_grad_(True)

        perturbed_data.x.requires_grad = True

        perturbed_data.edge_index = perturbed_data.edge_index.float()
        perturbed_data.edge_index.requires_grad = True

        # Forward pass with dense adjacency
        output = model(perturbed_data)['logits']
        loss = criterion(output, data.y)
        loss.backward()

        grad = A.grad
        grad[torch.arange(data.num_nodes), torch.arange(data.num_nodes)] = -float('inf')  # Ignore self-loops

        # Determine edge flip
        max_val, max_idx = torch.max(torch.abs(grad), dim=1)
        row = torch.argmax(max_val)
        col = max_idx[row]

        current_A = A.clone().detach()
        if current_A[row, col] == 0:
            perturbed_data.edge_index = torch.cat([perturbed_data.edge_index, torch.tensor([[row], [col]])], dim=1)
        else:
            mask = ~((perturbed_data.edge_index[0] == row) & (perturbed_data.edge_index[1] == col))
            perturbed_data.edge_index = perturbed_data.edge_index[:, mask]

    return perturbed_data


def attack_2(data, model, criterion):

    perturbed_data = adversarial_attack(data, model, criterion, budget=5)

    with torch.no_grad():
        attack_probs = model(perturbed_data)['probs']

        return F.softmax(attack_probs).argmax(dim=1).item()