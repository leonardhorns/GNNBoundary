import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import networkx as nx
import copy
import secrets
import os
import pickle
import glob
import torch.nn.functional as F
import torch_geometric as pyg
import time

# TODO: refactor
# from .datasets import *


class Trainer:
    def __init__(self,
                 sampler,
                 discriminator,
                 criterion,
                 optim_factory,
                 dataset,
                 budget_penalty=None,):
        self.sampler = sampler
        self.discriminator = discriminator
        self.criterion = criterion
        self.budget_penalty = budget_penalty
        self.optim_factory = optim_factory
        self.dataset = dataset
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator.to(self.device)
        self.init()

    def init(self):
        self.sampler.init()
        self.iteration = 0
        self.optimizer, self.scheduler = self.optim_factory(self.sampler)

    def train(self, iterations,
              show_progress=True,
              target_probs: dict[int, tuple[float, float]] = None,
              target_size=None,
              w_budget_init=1,
              w_budget_inc=1.05,
              w_budget_dec=0.99,
              k_samples=32):
        # self.bkup_state = copy.deepcopy(self.sampler.state_dict())
        # self.bkup_criterion = copy.deepcopy(self.criterion)
        # self.bkup_iteration = self.iteration
        self.discriminator.eval()
        self.sampler.train()
        budget_penalty_weight = w_budget_init

        logs = {
            'cls_probs': [],
            'nan_samples': [],
        }
        for _ in (bar := tqdm(
            range(int(iterations)),
            initial=self.iteration,
            total=self.iteration+iterations,
            disable=not show_progress
        )):
            for opt in self.optimizer:
                opt.zero_grad()

            cont_data, disc_data = self.sampler(k=k_samples, mode='both')
            cont_data.to(self.device)
            disc_data.to(self.device)
            # TODO: potential bug
            cont_out = self.discriminator(cont_data, edge_weight=cont_data.edge_weight)
            disc_out = self.discriminator(disc_data, edge_weight=disc_data.edge_weight)
            
            disc_props = disc_out["probs"][~disc_out["probs"].isnan().any(dim=1)]
            logs["nan_samples"].append(k_samples - disc_props.shape[0])
            if disc_props.shape[0] != k_samples:
                print(self.sampler.expected_m)
                return False, logs

            expected_probs = disc_props.mean(dim=0)
            logs['cls_probs'].append(expected_probs.detach().cpu())

            if target_probs and all([
                min_p <= expected_probs[classes].item() <= max_p
                for classes, (min_p, max_p) in target_probs.items()
            ]):
                if target_size and self.sampler.expected_m <= target_size:
                    logs['final_probs'] = expected_probs.detach().cpu()
                    break
                budget_penalty_weight *= w_budget_inc
            else:
                budget_penalty_weight = max(w_budget_init, budget_penalty_weight * w_budget_dec)

            loss = self.criterion(cont_out | self.sampler.to_dict())
            if self.budget_penalty:
                loss += self.budget_penalty(self.sampler.theta) * budget_penalty_weight
            loss.backward()  # Back-propagate gradients
            #torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), 1.0)

            for opt in self.optimizer:
                opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            size = self.sampler.expected_m
            scores = disc_out["logits"].mean(axis=0).tolist()
            score_dict = {v: scores[k] for k, v in self.dataset.GRAPH_CLS.items()}
            penalty_weight = {'bpw': budget_penalty_weight} if self.budget_penalty else {}
            bar.set_postfix({'size': size} | penalty_weight | score_dict)
            # print(f"{iteration=}, loss={loss.item():.2f}, {size=}, scores={score_dict}")
            self.iteration += 1
        else:
            return False, logs
        return True, logs

    @torch.no_grad()
    def predict(self, G):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        batch = batch.to(self.device)
        return self.discriminator(batch)
    
    @torch.no_grad()
    def predict_batch(self, Gs):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True) for G in Gs])
        batch = batch.to(self.device)
        return self.discriminator(batch)

    @torch.no_grad()
    def quantatitive(self, sample_size=1000, sample_fn=None, use_train_sampling=False, return_model_out=False):
        if use_train_sampling:
            self.sampler.eval()
            samples = self.sampler(k=sample_size, mode='discrete').to(self.device)
            if return_model_out:
                return self.discriminator(samples, edge_weight=samples.edge_weight)
            probs = self.discriminator(samples, edge_weight=samples.edge_weight)["probs"]
        else:
            sample_fn = sample_fn or (lambda: self.evaluate(bernoulli=True))
            graphs = [sample_fn() for _ in range(sample_size)]
            probs = self.predict_batch(graphs)["probs"]
        return dict(label=list(self.dataset.GRAPH_CLS.values()),
                    mean=probs.mean(dim=0),
                    std=probs.std(dim=0))

    @torch.no_grad()
    def quantatitive_baseline(self, **kwargs):
        return self.quantatitive(sample_fn=lambda: nx.gnp_random_graph(n=self.sampler.n, p=1/self.sampler.n),
                                 **kwargs)

    # TODO: do not rely on dataset for drawing
    @torch.no_grad()
    def evaluate(self, *args, show=False, connected=False, **kwargs):
        self.sampler.eval()
        G = self.sampler.sample(*args, **kwargs)
        if connected:
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
        if show:
            self.show(G)
            plt.show()
        return G

    def show(self, G, ax=None):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        pred = self.predict(G)
        logits = pred["logits"].mean(dim=0).tolist()
        probs = pred["probs"].mean(dim=0).tolist()
        print(f"{n=} {m=}")
        print(f"{logits=}")
        print(f"{probs=}")
        self.dataset.draw(G, ax=ax)

    def save_graph(self, G, cls_idx, root="results"):
        if isinstance(cls_idx, tuple):
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx[0]]}-{self.dataset.GRAPH_CLS[cls_idx[1]]}"
        else:
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx]}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        pickle.dump(G, open(f"{path}/{name}.pkl", "wb"))
        self.show(G)
        plt.savefig(f"{path}/{name}.png", bbox_inches="tight")
        plt.show()

    def load_graph(self, id, root="results"):
        path = f"{root}/{self.dataset.name}/*"
        G = pickle.load(open(glob.glob(f"{path}/{id}.pkl")[0], "rb"))
        self.show(G)
        return G

    def save_sampler(self, cls_idx, root="sampler_ckpts", retrain=False):
        if isinstance(cls_idx, int):
            path = f"{root}/{self.dataset.name}{f'_retrained' if retrain else ''}/{cls_idx}"
        else:
            path = f"{root}/{self.dataset.name}{f'_retrained' if retrain else ''}/{'-'.join(map(str, cls_idx))}"
        name = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(path, exist_ok=True)
        self.sampler.save(f"{path}/{name}.pt")
        return f"{path}/{name}.pt"

    def batch_generate(self, cls_idx, total, num_boundary_samples=0, show_runs=False, retrain=False, **train_args):
        pbar = tqdm(total=total)
        count = 0
        logs = []
        while count < total:
            self.init()
            converged, run_logs = self.train(**train_args)
            run_logs["converged"] = converged

            if show_runs:
                cls_probs = torch.stack(run_logs['cls_probs']).T

                for cls in cls_idx:
                    plt.plot(cls_probs[cls], label=f"cls {cls}")
                plt.legend()
                plt.ylim(0, 1)
                plt.savefig(f"plots/quantitative/{count}.png", bbox_inches="tight", dpi=100)
                plt.show()
            del run_logs['cls_probs']

            if converged:
                save_path = self.save_sampler(cls_idx, retrain=retrain)
                run_logs["save_path"] = save_path

                if num_boundary_samples > 0:
                    idx = list(cls_idx)
                    run_logs["train_scores"] = self.quantatitive(sample_size=num_boundary_samples, use_train_sampling=True)
                    run_logs["eval_scores"] = self.quantatitive(sample_size=num_boundary_samples)

                    if show_runs:
                        print("final", run_logs["final_probs"][idx].tolist())
                        print("train", run_logs["train_scores"]["mean"][idx].tolist())
                        print("eval", run_logs["eval_scores"]["mean"][idx].tolist())

            logs.append(run_logs)
            count += 1
            pbar.update(1)
        return logs

    def get_training_success_rate(self, total, epochs, show_progress=False):
        iters = []
        for _ in (bar := trange(total)):
            self.init()
            if self.train(epochs, show_progress=show_progress):
                iters.append(self.iteration)
            bar.set_postfix({'count': len(iters)})
        return iters
