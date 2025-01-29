import gnnboundary
import torch
import json
import time
import os
import numpy as np


def get_dataset_setup(
        dataset_name,
        use_gat=False,
        retrained=False
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
    complex_ckpts = {"collab_retrained", "motif_retrained", "binary_redit"}
    ckpt = torch.load(path)['model'] if dataset_name in complex_ckpts else torch.load(path)
    model.load_state_dict(ckpt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    return dataset, model


def get_trainer(cls_idx, dataset_name, use_gat=False, use_retrained=False, sampler_path=None):
    dataset, model = get_dataset_setup(dataset_name, use_gat=use_gat, retrained=use_retrained)

    sampler = gnnboundary.GraphSampler(
        max_nodes=50, #430
        temperature=0.15,
        num_node_cls=len(dataset.NODE_CLS),
        learn_node_feat=True,
    )

    criterion = gnnboundary.WeightedCriterion([
        dict(key="logits", criterion=gnnboundary.DynamicBalancingBoundaryCriterion(
            classes=list(cls_idx), alpha=1, beta=1
        ), weight=25),
        # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
        # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
        # dict(key="logits", criterion=gnnboundary.MeanPenalty(), weight=1),
        dict(key="omega", criterion=gnnboundary.NormPenalty(order=1), weight=1),
        dict(key="omega", criterion=gnnboundary.NormPenalty(order=2), weight=1),
        # dict(key="xi", criterion=NormPenalty(order=1), weight=0),
        # dict(key="xi", criterion=NormPenalty(order=2), weight=0),
        # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
        # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
        # dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
    ])

    def get_optimizer(sampler):
        optimizer = torch.optim.SGD(sampler.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1, total_steps=500)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], scheduler

    trainer = gnnboundary.Trainer(
        sampler=sampler,
        discriminator=model,
        criterion=criterion,
        optim_factory=get_optimizer,
        dataset=dataset,
        budget_penalty=gnnboundary.BudgetPenalty(budget=30, order=2, beta=1),
    )
    
    if sampler_path is not None:
        sampler.load(sampler_path)
    return trainer


def train_eval(cls_idx, dataset_name, num_runs, num_samples, train_args, use_gat=False, use_retrained=False, show_runs=False):
    start = time.time()
    
    train_args["target_probs"] = {cls_idx[0]: train_args["target_probs"], cls_idx[1]: train_args["target_probs"]}
    trainer = get_trainer(cls_idx, dataset_name, use_gat=use_gat, use_retrained=use_retrained)
    logs = trainer.batch_generate(cls_idx, total=num_runs, num_boundary_samples=num_samples, show_runs=show_runs, **train_args)
    
    converged = [(l["train_scores"], l["eval_scores"]) for l in logs if l["converged"]]
    scores = {}
    
    for label, score_list in zip(["train", "eval"], zip(*converged)):
        means = torch.stack([s["mean"] for s in score_list])
        stds = torch.stack([s["std"] for s in score_list])
        
        loss = (means[:, list(cls_idx)] - 0.5).abs() + stds[:, list(cls_idx)]
        best_idx = loss.sum(dim=1).argmin().item()
        
        scores[label] = {"mean_mean": means.mean(dim=0).tolist(),
                         "mean_std": stds.mean(dim=0).tolist(),
                         "best_idx": best_idx,
                         "best_mean": means[best_idx].tolist(),
                         "best_std": stds[best_idx].tolist()}
     
    convergence_rate = len(converged) / num_runs
    scores["convergence_rate"] = convergence_rate
    total_time = time.time() - start
    scores["time"] = total_time
    
    print(f"Time: {total_time} seconds")
    print(f"Classes: {cls_idx}", f"Num runs: {num_runs}, num samples: {num_samples}", sep="\n", end="\n\n")
    print(f"Convergence rate: {convergence_rate}")
    if len(converged) > 0:
        print(f"""Train - mean: {scores["train"]["mean_mean"]}, std: {scores["train"]["mean_std"]}
            best_idx: {scores["train"]["best_idx"]},
            best: {scores["train"]["best_mean"]}, std: {scores["train"]["best_std"]}""")
        print(f"""Eval - mean: {scores["eval"]["mean_mean"]}, std: {scores["eval"]["mean_std"]}
            best_idx: {scores["eval"]["best_idx"]},
            best: {scores["eval"]["best_mean"]}, std: {scores["eval"]["best_std"]}""")
    
    return scores, logs


train_args = dict(
    iterations=500,
    target_probs=(0.45, 0.55),
    show_progress=True,
    target_size=60,
    w_budget_init=1,
    w_budget_inc=1.15,
    w_budget_dec=0.98,
    k_samples=32,
)


def get_result_paths(dataset_name, cls_idx, save_dir, use_json=False):
    os.makedirs(save_dir, exist_ok=True)
    ext = "json" if use_json else "pt"
    
    base_name = f"{dataset_name}_{cls_idx[0]}-{cls_idx[1]}"
    return os.path.join(save_dir, f"{base_name}_scores.{ext}"), os.path.join(save_dir, f"{base_name}_logs.{ext}")


cls_idx = (0, 3)
dataset_name = 'motif'
save_dir= "sampler_ckpts/GAT"

scores, logs = train_eval(
    cls_idx,
    dataset_name,
    num_runs=1000,
    num_samples=500,
    show_runs=True,
    use_gat=True,
    train_args=train_args,
)
scores_path, logs_path = get_result_paths(dataset_name, cls_idx, save_dir)
torch.save(scores, scores_path)
torch.save(logs, logs_path)
