#!/usr/bin/env python

import argparse
import os
import json
import torch
from pprint import pprint

import gnnboundary
from experiments.gnnboundary import train_eval, evaluate_sampler, baseline, calculate_adjacency_matrix


hyperparams = {
    "default": {
        "max_nodes": 25,
        "lr": 1.,
        "budget": 10,
        "target_size": 60,
    },
    "gat": {
        "max_nodes": 50,
        "lr": 0.01,
        "budget": 30,
        "target_size": 60,
    },
    "reddit": {
        "max_nodes": 150,
        "lr": 1.,
        "budget": 100,
        "target_size": 400,
    },
}

experiment_configurations = {
    "motif": dict(
        dataset_name="motif",
        class_pairs=[(0, 1), (0, 2), (1, 3)],
        hparams="default",
    ),
    "collab": dict(
        dataset_name="collab",
        class_pairs=[(0, 1), (0, 2), (1, 2)],
        hparams="default",
    ),
    "enzymes": dict(
        dataset_name="enzymes",
        class_pairs=[(0, 5), (1, 2), (1, 5)],
        hparams="default",
    ),
    "binary_reddit": dict(
        dataset_name="binary_reddit",
        class_pairs=[(0, 1)],
        hparams="reddit",
    ),
    "motif_gat": dict(
        dataset_name="motif",
        class_pairs=[(0, 1), (0, 3), (1, 2), (1, 3)],
        hparams="gat",
        use_gat=True,
    ),
}


def run_experiment(experiment_name, dataset_name, class_pairs, hparams, num_runs=1000, use_gat=False):
    # Adjacency Matrix
    print("Computing adjacency matrix...")
    calculate_adjacency_matrix(dataset_name, experiment_name)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # GNNBoundary Training
    hyperparameters = hyperparams[hparams]
    scores = {cls_idx: {} for cls_idx in class_pairs}
    print("Training GNNBoundary...")
    for cls_idx in class_pairs:
        pair_scores, logs = train_eval(cls_idx,
                                       dataset_name,
                                       num_runs=num_runs,
                                       hparams=hyperparameters,
                                       use_gat=use_gat,
                                       use_retrained=False,
                                       show_runs=False)
        scores[cls_idx] |= pair_scores
        torch.save(logs, f"results/{experiment_name}_{pair_scores['class_pair']}_logs.pth")
    
    # Boundary evaluation
    converged_pairs = [cls_idx for cls_idx, pair_scores in scores.items() if pair_scores["convergence_rate"] > 0]
    print("Running boundary evaluation...")
    evaluation = evaluate_sampler(converged_pairs,
                                  dataset_name,
                                  num_samples=500,
                                  hparams=hyperparameters,
                                  experiment_name=experiment_name,
                                  use_gat=use_gat,
                                  from_retrained_model=False,
                                  sampler_ckpt_paths=[scores[cls_idx]["sampler_ckpt"] for cls_idx in converged_pairs],
                                  save_path=os.path.join("figures", experiment_name))
    
    for cls_idx, complexity in evaluation['boundary_complexity'].items():
        scores[cls_idx]["complexity"] = complexity
    
    # Baseline
    print("Running baseline evaluation...")
    baseline_scores = baseline(class_pairs, dataset_name, num_samples=500)
    for cls_idx, s in baseline_scores.items():
        scores[cls_idx] |= s
    
    json_scores = {s["class_pair"]: s for s in scores.values()}
    with open(f"results/{experiment_name}_scores.json", "w") as f:
        json.dump(json_scores, f, indent=4)
    
    print("Experiment completed. Results:", end="\n\n")
    pprint(json_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Name of the experiment to run")
    args = parser.parse_args()

    if args.experiment not in experiment_configurations:
        raise ValueError(f"Unknown experiment name: {args.experiment}. Choose from {list(experiment_configurations.keys())}")
    
    run_experiment(args.experiment, **experiment_configurations[args.experiment], num_runs=5)
