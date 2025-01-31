import gnnboundary
import torch
import json
import time
import os
import numpy as np
import random
from tqdm import tqdm

def get_dataset_setup(
        dataset_name,
        use_gat=False,
        retrained=False,
        model_path=None,
    ):

    datasets = {
        "motif": gnnboundary.MotifDataset,
        "collab": gnnboundary.CollabDataset,
        "enzymes": gnnboundary.ENZYMESDataset,
        "binary_reddit": gnnboundary.RedditDataset,
    }
    models = {
        "motif": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=6,
            num_layers=3,
        ), "ckpts/motif.pt"),
        "collab": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=64,
            num_layers=5,
        ), "ckpts/collab.pt"),
        "enzymes":lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(dataset.NODE_CLS),
            num_classes=len(dataset.GRAPH_CLS),
            hidden_channels=32,
            num_layers=3,
        ), "ckpts/enzymes.pt"),
        "binary_reddit":lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(dataset.NODE_CLS),
            num_classes=len(dataset.GRAPH_CLS),
            hidden_channels=64,
            num_layers=5,
        ), "ckpts/reddit.pt"),
        "motif_gat": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=6,
            num_layers=3,
            use_gat=True,
        ), "ckpts/motif_gat.pt"),
        "collab_gat": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=64,
            num_layers=5,
            use_gat=True,
        ), "ckpts/collab_gat.pt"),
        "motif_retrained": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=6,
            num_layers=3,
        ), "ckpts/motif_retrained.pt"),
        "collab_retrained": lambda ds: (gnnboundary.GCNClassifier(
            node_features=len(ds.NODE_CLS),
            num_classes=len(ds.GRAPH_CLS),
            hidden_channels=64,
            num_layers=5,
        ), "ckpts/collab_retrained.pt"),
    }

    dataset = datasets[dataset_name](seed=12345)
    if use_gat:
        dataset_name += "_gat"
    if retrained:
        dataset_name += "_retrained"

    model, path = models[dataset_name](dataset)
    if model_path is not None:
        path = model_path
    complex_ckpts = {"collab_retrained", "motif_retrained", "binary_redit"}
    ckpt = torch.load(path)['model'] if dataset_name in complex_ckpts else torch.load(path)
    model.load_state_dict(ckpt)

    return dataset, model


def get_trainer(cls_idx, dataset_name, hparams, use_gat=False, use_retrained=False, sampler_path=None, dataset=None, model=None):
    if dataset is not None and model is not None:
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
    else:
        dataset, model = get_dataset_setup(dataset_name, use_gat=use_gat, retrained=use_retrained)
    
    sampler = gnnboundary.GraphSampler(
        max_nodes=hparams["max_nodes"],
        temperature=0.15,
        num_node_cls=len(dataset.NODE_CLS),
        learn_node_feat=True,
    )

    criterion = gnnboundary.WeightedCriterion([
        dict(key="logits", criterion=gnnboundary.DynamicBalancingBoundaryCriterion(
            classes=list(cls_idx), alpha=1, beta=1
        ), weight=25),
        dict(key="omega", criterion=gnnboundary.NormPenalty(order=1), weight=1),
        dict(key="omega", criterion=gnnboundary.NormPenalty(order=2), weight=1),
    ])

    def get_optimizer(sampler):
        optimizer = torch.optim.SGD(sampler.parameters(), lr=hparams["lr"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        return [optimizer], scheduler

    trainer = gnnboundary.Trainer(
        sampler=sampler,
        discriminator=model,
        criterion=criterion,
        optim_factory=get_optimizer,
        dataset=dataset,
        budget_penalty=gnnboundary.BudgetPenalty(budget=hparams["budget"], order=2, beta=1),
    )
    
    if sampler_path is not None:
        sampler.load(sampler_path)
    return trainer


def train_eval(cls_idx, dataset_name, num_runs, hparams, use_gat=False, use_retrained=False, show_runs=False):
    start_time = time.time()
    
    # GNNBoundary Training
    train_args = dict(
        iterations=500,
        target_probs={cls_idx[0]: (0.45, 0.55), cls_idx[1]: (0.45, 0.55)},
        show_progress=False,
        target_size=hparams["target_size"],
        w_budget_init=1,
        w_budget_inc=1.15,
        w_budget_dec=0.98,
        k_samples=32,
    )
    
    trainer = get_trainer(cls_idx, dataset_name, hparams, use_gat=use_gat, use_retrained=use_retrained)
    logs = trainer.batch_generate(cls_idx, total=num_runs, num_boundary_samples=500, show_runs=show_runs, **train_args)
    scores = {"class_pair": f"{trainer.dataset.GRAPH_CLS[cls_idx[0]]}_{trainer.dataset.GRAPH_CLS[cls_idx[1]]}"}
    
    converged = [l for l in logs if l["converged"]]
    if len(converged) > 0:
        means = torch.stack([l["train_scores"]["mean"] for l in converged])
        stds = torch.stack([l["train_scores"]["std"] for l in converged])
        
        loss = (means[:, list(cls_idx)] - 0.5).abs() + stds[:, list(cls_idx)]
        best_idx = loss.sum(dim=1).argmin().item()
        
        means = means[:, list(cls_idx)]
        stds = stds[:, list(cls_idx)]
        scores |= {
            "best_idx": best_idx,
            "best_mean": means[best_idx].tolist(),
            "best_std": stds[best_idx].tolist(),
            "means_mean": means.mean(dim=0).tolist(),
            "means_std": means.std(dim=0).tolist(),
            "stds_mean": stds.mean(dim=0).tolist(),
            "stds_std": stds.std(dim=0).tolist(),
        }
        
        sampler_ckpt = converged[best_idx]["save_path"]
        scores["sampler_ckpt"] = sampler_ckpt
    
    convergence_rate = len(converged) / num_runs
    scores["convergence_rate"] = convergence_rate
    
    total_time = time.time() - start_time
    scores["time"] = total_time
    return scores, logs


def evaluate_sampler(adjacent_class_pairs,
                     dataset_name,
                     num_samples,
                     hparams,
                     experiment_name,
                     use_gat=False,
                     from_retrained_model=False,
                     sampler_ckpt_paths=[],
                     save_path=None):

    #make sure that order of sampler_ckpt_paths is the same as adjacent class pairs
    trainers = []
    
    dataset, model = get_dataset_setup(dataset_name, use_gat=use_gat, retrained=from_retrained_model)
    
    for sampler_path, class_pair in zip(sampler_ckpt_paths, adjacent_class_pairs):
        trainers.append(get_trainer(class_pair, dataset_name=dataset_name, hparams=hparams, sampler_path=sampler_path, dataset=dataset, model=model))

    save_path = save_path or f'./figures/{dataset_name}'
    evaluation = gnnboundary.evaluate_boundary(dataset,
                                               trainers,
                                               adjacent_class_pairs,
                                               model,
                                               num_samples,
                                               experiment_name)

    evaluation['boundary_margin'][evaluation['boundary_margin'] == 0] = np.nan
    evaluation['boundary_thickness'][evaluation['boundary_thickness'] == 0] = np.nan

    annot_size, label_size = 30, 25

    gnnboundary.draw_matrix(
        dataset.model_evaluate(model)['cm'],
        dataset.GRAPH_CLS.values(),
        file_name=f'{experiment_name}_cm.png',
        save_path=save_path,
        fmt='d',
        annotsize=annot_size,
        labelsize=label_size,
        show_plot=False,
    )
    gnnboundary.draw_matrix(
        evaluation['boundary_margin'],
        dataset.GRAPH_CLS.values(),
        xlabel='Decision boundary',
        ylabel='Decision region',
        file_name=f'{experiment_name}_boundary_margin.png',
        save_path=save_path,
        annotsize=annot_size,
        labelsize=label_size,
        show_plot=False,
    )
    gnnboundary.draw_matrix(
        evaluation['boundary_thickness'],
        dataset.GRAPH_CLS.values(),
        xlabel='Decision boundary',
        ylabel='Decision region',
        file_name=f'{experiment_name}_boundary_thickness.png',
        save_path=save_path,
        annotsize=annot_size,
        labelsize=label_size,
        show_plot=False,
    )

    return evaluation


def baseline(cls_pairs, dataset_name, num_samples=500, use_gat=False, retrained=False):
    dataset, model = get_dataset_setup(dataset_name, use_gat=use_gat, retrained=retrained)

    baseline_scores = {cls_idx: {} for cls_idx in cls_pairs}
    for cls_idx in tqdm(cls_pairs):
        generator = gnnboundary.utils.BaselineGenerator(dataset.split_by_class(), cls_idx)
        samples = generator.sample(num_samples)

        model.eval()
        probs = model.forward(dataset.convert(samples))['probs'][:, list(cls_idx)]
        baseline_scores[cls_idx]["baseline_mean"] = probs.mean(dim=0)
        baseline_scores[cls_idx]["baseline_std"] = probs.std(dim=0)
    
    return baseline_scores


def calculate_adjacency_matrix(dataset_name, experiment_name, use_gat=False, retrained=False):
    experiment_results = []
    dataset, model = get_dataset_setup(dataset_name, use_gat=use_gat, retrained=retrained)
    dataset_list_pred = dataset.split_by_pred(model)
    
    for _ in tqdm(range(10)):
        adj_ratio_mat, _ = gnnboundary.pairwise_boundary_analysis(model, dataset_list_pred)
        experiment_results.append(adj_ratio_mat)

    result = np.array(experiment_results).mean(axis=0)
    gnnboundary.draw_matrix(
        result, 
        names=dataset.GRAPH_CLS.values(),
        fmt='.2f',
        file_name=f"{experiment_name}_adjacency.jpg",
        save_path="figures",
        show_plot=False)
