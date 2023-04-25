import sys
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import fmin, fminbound
from scipy.stats import norm
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics import Metric
import plotly.express as px
from torchmetrics.classification import BinaryMatthewsCorrCoef, BinaryAUROC


def find_intersect(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    def pdf1(x):
        return norm.pdf(x, mu1, sigma1)

    def pdf2(x):
        return norm.pdf(x, mu2, sigma2)

    if pdf1(mu1) < pdf2(mu1) or pdf1(mu2) > pdf2(mu2):
        # print(f"{pdf1(mu1):.4f} | {pdf2(mu1):.4f} || {pdf1(mu2):.4f} | {pdf2(mu2):.4f}")
        # return float("nan")
        pass

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    mid = float(fminbound(func=lambda x: abs(pdf1(x) - pdf2(x)), x1=min(mu1, mu2), x2=max(mu1, mu2)))
    sys.stdout = save_stdout

    # x_axis = np.arange(0, 1, 0.001)
    # plt.plot(x_axis, norm.pdf(x_axis, mu1, sigma1))
    # plt.plot(x_axis, norm.pdf(x_axis, mu2, sigma2))
    # plt.vlines(mid, ymin=0, ymax=max(pdf1(mu1), pdf2(mu1), pdf1(mu2), pdf2(mu2)), color="r")
    # plt.show()
    # print(abs(pdf1(mid) - pdf2(mid)))
    # print(abs(pdf1((mu1 + mu2) / 2) - pdf2((mu1 + mu2) / 2)))
    return mid


class DistOverlap(Metric):
    """A metric keeping track of the distributions of predicted values for positive and negative samples"""

    def __init__(self, prefix="", **kwargs):
        super(DistOverlap, self).__init__()
        self.add_state("pos", default=[], dist_reduce_fx="cat")
        self.add_state("neg", default=[], dist_reduce_fx="cat")
        if prefix != "":
            self.prefix = ""
        else:
            self.prefix = prefix + "_"

    def __name__(self):
        return self.prefix + "DistOverlap"

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Store the predictions separated into those of positive samples and those of negative samples"""
        pos = preds[target == 1]
        neg = preds[target == 0]
        self.pos += pos
        self.neg += neg

    def compute_params(self):
        self.pos = torch.stack(self.pos)
        self.neg = torch.stack(self.neg)
        pos_mu, pos_sigma = torch.mean(self.pos), torch.std(self.pos)
        neg_mu, neg_sigma = torch.mean(self.neg), torch.std(self.neg)
        return (pos_mu, pos_sigma), (neg_mu, neg_sigma)

    def compute(self) -> Any:
        """Calculate the metric based on the samples from the update rounds"""
        if len(self.pos) == 0 or len(self.neg) == 0:
            return torch.nan

        (pos_mu, pos_sigma), (neg_mu, neg_sigma) = self.compute_params()

        return torch.tensor(
            NormalDist(mu=pos_mu.item(), sigma=pos_sigma.item()).overlap(
                NormalDist(mu=neg_mu.item(), sigma=neg_sigma.item())
            )
        )


class EmbeddingMetric(Metric):
    def __init__(self, classes, prefix=""):
        super(EmbeddingMetric, self).__init__()
        self.classes = classes
        self.add_state("embeds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        if prefix != "":
            self.prefix = ""
        else:
            self.prefix = prefix + "_"

    def __name__(self):
        return self.prefix + "DistOverlap"

    def update(self, embeddings, labels) -> None:
        self.embeds += embeddings
        self.labels += labels

    def compute(self) -> Any:
        self.embeds = torch.stack(self.embeds).cpu().numpy()
        self.labels = torch.stack(self.labels).cpu().numpy()
        tsne_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
            self.embeds
        )
        for i, c in enumerate(self.classes):
            plt.scatter(tsne_embeds[self.labels == i, 0], tsne_embeds[self.labels == i, 1], label=c, s=20)
        embed_df = pd.DataFrame(tsne_embeds)
        embed_df["size"] = 50
        embed_df["type"] = list(map(lambda x: self.classes[x], self.labels))
        return px.scatter(
            embed_df,
            0,
            1,
            color="type",
            symbol="type",
            opacity=0.5,
            width=400,
            height=400,
            size="size",
            title="Embedding tSNE",
        )


class DyBiAUROC(Metric):
    def __init__(self, *args, **kwargs):
        super(DyBiAUROC, self).__init__()
        if "thresholds" not in kwargs:
            kwargs["thresholds"] = torch.tensor(0.5).float().unsqueeze(0).to(self.device)
        self.bin_auroc = BinaryAUROC(**kwargs)
        self.dist = DistOverlap(**kwargs)

    @staticmethod
    def __name__():
        return "DynamicBinaryAUROC"

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.bin_auroc.update(preds, target)
        self.dist.update(preds, target)

    def compute(self):
        (pos_mu, pos_sigma), (neg_mu, neg_sigma) = self.dist.compute_params()
        threshold = find_intersect(pos_mu.item(), pos_sigma.item(), neg_mu.item(), neg_sigma.item())

        if threshold != threshold:
            return torch.tensor(torch.nan)

        self.bin_auroc.thresholds = torch.tensor(threshold).float().unsqueeze(0).to(self.device)
        return self.bin_auroc.compute()

    def reset(self):
        self.bin_auroc.reset()
        self.dist.reset()


class DyBiMCC(Metric):
    def __init__(self, *args, **kwargs):
        super(DyBiMCC, self).__init__()
        if "threshold" not in kwargs:
            kwargs["threshold"] = 0.5
        self.bin_mcc = BinaryMatthewsCorrCoef(**kwargs)
        self.dist = DistOverlap(**kwargs)

    @staticmethod
    def __name__():
        return "DynamicMCC"

    def update(self, preds, targets):
        self.bin_mcc.update(preds, targets)
        self.dist.update(preds, targets)

    def compute(self):
        (pos_mu, pos_sigma), (neg_mu, neg_sigma) = self.dist.compute_params()
        threshold = find_intersect(pos_mu.item(), pos_sigma.item(), neg_mu.item(), neg_sigma.item())

        if threshold != threshold:
            return torch.tensor(torch.nan)

        self.bin_mcc.threshold = threshold
        return self.bin_mcc.compute()

    def reset(self):
        self.bin_mcc.reset()
        self.dist.reset()


class Decider(Metric):
    def __init__(self, **kwargs):
        super(Decider, self).__init__()
        self.dist = DistOverlap(**kwargs)

    @staticmethod
    def __name__():
        return "Decider"

    def update(self, preds, targets):
        self.dist.update(preds, targets)

    def compute(self):
        (pos_mu, pos_sigma), (neg_mu, neg_sigma) = self.dist.compute_params()
        # print(f"{pos_mu.item():.4f} | {pos_sigma.item():.4f} || {neg_mu.item():.4f} | {neg_sigma.item():.4f} => {find_intersect(pos_mu.item(), pos_sigma.item(), neg_mu.item(), neg_sigma.item())}")
        return find_intersect(pos_mu.item(), pos_sigma.item(), neg_mu.item(), neg_sigma.item())

    def reset(self):
        self.dist.reset()


if __name__ == '__main__':
    print(find_intersect(0.4999, 0.0010, 0.5002, 0.0010))
    print(find_intersect(0.6117, 0.0266, 0.6024, 0.0352))
