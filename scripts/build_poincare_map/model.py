# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics.pairwise import pairwise_distances
from torch.autograd import Function
from torch import nn
import numpy as np
import torch

eps = 1e-5
boundary = 1 - eps

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)


def poincare_root(opt, labels, features):
    if opt.root is not None:
        head_idx = np.where(labels == opt.root)[0]

        if len(head_idx) > 1:
            # medoids in Euclidean space
            D = pairwise_distances(features[head_idx, :], metric='euclidean')
            return head_idx[np.argmin(D.mean(axis=0))]
        else:
            return head_idx[0]

    return -1


def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)        
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
            torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):  
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):    
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

    
def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


class PoincareEmbedding(nn.Module):
    def __init__(self,
                 size,
                 dim,
                 dist=PoincareDistance,
                 max_norm=1,
                 Qdist='laplace',
                 lossfn='klSym',
                 gamma=1.0,
                 cuda=0):
        super(PoincareEmbedding, self).__init__()

        self.dim = dim
        self.size = size
        self.lt = nn.Embedding(size, dim, max_norm=max_norm)

        ## pour ajout de points : initialiser ici avec les poids de l'ancien embedding ?
        self.lt.weight.data.uniform_(-1e-4, 1e-4)
        #####

        self.dist = dist
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == 'kl':
            self.lossfn = nn.KLDivLoss()
        elif lossfn == 'klSym':
            self.lossfn = klSym
        elif lossfn == 'mse':
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()

    def forward(self, inputs):
        embs_all = self.lt.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.lt(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)        

        if self.lossfnname == 'kl':
            if self.Qdist == 'laplace':
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'klSym':
            if self.Qdist == 'laplace':
                return self.sm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'mse':
            return self.sm(-self.gamma * dists)
        else:
            raise NotImplementedError

    def infer_embedding_for_point(self,
                                  target,
                                  n_steps: int = 200,
                                  lr: float = 0.1,
                                  init: str = 'random',
                                  device: str = None):
        """
        Infers an embedding vector for a single new point given its target
        similarity/distribution to the existing corpus.

        This routine keeps the existing embeddings fixed and optimizes a
        single new embedding so that its predicted distribution (Q) matches
        the provided `target` distribution (length == current size).

        Args:
            target: 1D array-like (length == self.size) or torch tensor with
                similarity/probability scores between the new point and each
                existing item. The method will normalize it to sum to 1.
            n_steps: number of optimization steps.
            lr: learning rate for the per-point optimizer.
            init: 'random' (uniform small) or 'zeros'.
            device: torch device string, if None uses model parameters device.

        Returns:
            numpy array of shape (dim,) with the inferred embedding (inside
            the Poincaré ball, i.e. norm < 1).

        Notes:
            - This does not modify the model's existing embedding table.
            - The new point is optimized alone, using the current embeddings as
              fixed anchors (detached). This is a cheap way to "add" a point
              to an existing projection without retraining everything.
        """
        # device selection
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        # prepare fixed existing embeddings (detached)
        with torch.no_grad():
            old_embs = self.lt.weight.detach().clone().to(device)

        # prepare target tensor (length == old_size)
        if not isinstance(target, torch.Tensor):
            target_t = torch.tensor(target, dtype=torch.float32, device=device)
        else:
            target_t = target.to(device).float()

        # Accept common 2D shapes (1, N) or (N, 1) by flattening to 1D.
        if target_t.ndim > 1:
            if target_t.shape[0] == 1:
                target_t = target_t.ravel()
            elif target_t.shape[1] == 1:
                target_t = target_t.ravel()

        if target_t.ndim != 1 or target_t.shape[0] != old_embs.shape[0]:
            raise ValueError(
                f"target must be 1D and length == {old_embs.shape[0]}; got shape {tuple(target_t.shape)}. "
                "If you computed target from distances, ensure it is a 1D array of length N (number of existing embeddings). "
                "Example: target = pairwise_distances(new_feat.reshape(1,-1), features).flatten()"
            )

        # normalize target to a probability vector
        if target_t.sum() <= 0:
            # avoid division by zero — use uniform small mass
            target_t = torch.ones_like(target_t) / float(target_t.shape[0])
        else:
            target_t = target_t / target_t.sum()

        # initialize new embedding parameter
        new = torch.zeros((1, self.dim), dtype=torch.float32, device=device)
        if init == 'random':
            new.uniform_(-1e-4, 1e-4)
        elif init == 'zeros':
            pass
        else:
            raise ValueError("init must be 'random' or 'zeros'")
        new = torch.nn.Parameter(new)

        optim = torch.optim.SGD([new], lr=lr)

        lossfn = self.lossfn

        for _ in range(n_steps):
            optim.zero_grad()

            # build batch shapes compatible with forward
            # embs_all : (1, old_size + 1, dim)
            embs_all = torch.cat([old_embs, new.detach()], dim=0).unsqueeze(0)

            # embs_inputs : start from new shape (1, dim) -> unsqueeze to (1,1,dim)
            # then expand to (1, old_size + 1, dim) to match embs_all
            embs_inputs = new.unsqueeze(0).expand(1, embs_all.size(1), self.dim)

            dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)  # (1, N+1)

            # we only have target for the existing points; append self-similarity
            # for the new point (last entry) as a small epsilon so shapes match
            pad = torch.tensor([1e-12], dtype=torch.float32, device=device)
            target_ext = torch.cat([target_t, pad], dim=0).unsqueeze(0)
            # renormalize
            target_ext = target_ext / target_ext.sum(dim=1, keepdim=True)

            # compute Q as in forward
            if self.lossfnname == 'kl':
                if self.Qdist == 'laplace':
                    preds = self.lsm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.lsm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.lsm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            elif self.lossfnname == 'klSym' or self.lossfnname == 'mse':
                if self.Qdist == 'laplace':
                    preds = self.sm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.sm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.sm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            loss = lossfn(preds, target_ext)
            loss.backward()

            # gradient step on the new embedding
            optim.step()

            # retract to Poincaré ball (ensure norm < 1 - eps)
            with torch.no_grad():
                norm = new.norm(p=2)
                if norm >= boundary:
                    new.mul_((boundary - 1e-8) / norm)

        return new.detach().cpu().numpy().reshape(self.dim,)
    
    def train_single_point(
        self,
        target,
        n_steps: int = 300,
        lr: float = 0.05,
        init: str = 'random',
        device: str = None,
        verbose: bool = False,
        k: int = 10,
        lambda_local: float = 1.0,
    ):
        """
        Infers a Poincaré embedding for a single new point while keeping
        existing embeddings fixed, with an explicit local attraction
        to nearest neighbors.

        Args:
            target: 1D array-like of length self.size
                    Similarity / probability distribution to existing points.
            n_steps: number of gradient steps.
            lr: learning rate.
            init: 'random' or 'zeros'.
            device: torch device (optional).
            verbose: print loss during optimization.
            k: number of nearest neighbors used for local attraction.
            lambda_local: weight of the local metric loss.

        Returns:
            new_embedding: numpy array of shape (dim,)
            losses: list of loss values
        """

        # device
        if device is None:
            device = next(self.parameters()).device

        # freeze existing embeddings
        with torch.no_grad():
            old_embs = self.lt.weight.detach().clone().to(device)

        # target distribution
        target = torch.tensor(target, dtype=torch.float32, device=device)
        if target.ndim != 1 or target.shape[0] != old_embs.shape[0]:
            raise ValueError(f"target must be 1D of length {old_embs.shape[0]}")

        if target.sum() <= 0:
            target = torch.ones_like(target) / target.numel()
        else:
            target = target / target.sum()

        # select top-k neighbors
        k = min(k, target.numel())
        topk = torch.topk(target, k=k).indices
        neighbor_embs = old_embs[topk]              # (k, dim)
        neighbor_w = target[topk]
        neighbor_w = neighbor_w / neighbor_w.sum()  # normalized weights

        # initialize new point
        new = torch.zeros((1, self.dim), device=device)
        if init == 'random':
            new.uniform_(-1e-4, 1e-4)
        elif init != 'zeros':
            raise ValueError("init must be 'random' or 'zeros'")

        new = torch.nn.Parameter(new)
        optimizer = torch.optim.SGD([new], lr=lr)

        losses = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # global distances (for KL term)
            embs_all = torch.cat([old_embs, new], dim=0).unsqueeze(0)
            embs_new = new.unsqueeze(0).expand_as(embs_all)
            dists = self.dist().apply(embs_new, embs_all).squeeze(-1)

            # target with self-distance padding
            pad = torch.tensor([1e-12], device=device)
            target_ext = torch.cat([target, pad]).unsqueeze(0)
            target_ext = target_ext / target_ext.sum(dim=1, keepdim=True)

            # predicted distribution (same logic as forward)
            if self.lossfnname == 'kl':
                if self.Qdist == 'laplace':
                    preds = self.lsm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.lsm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.lsm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            else:  # klSym or mse
                if self.Qdist == 'laplace':
                    preds = self.sm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.sm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.sm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError

            # global distribution loss
            loss_global = self.lossfn(preds, target_ext)

            # local metric attraction loss
            d_local = self.dist().apply(
                new.unsqueeze(0).expand(1, k, self.dim),
                neighbor_embs.unsqueeze(0)
            ).squeeze()

            loss_local = (neighbor_w * d_local).sum()

            # total loss
            loss = loss_global + lambda_local * loss_local
            loss.backward()
            optimizer.step()

            # retract into Poincaré ball
            with torch.no_grad():
                norm = new.norm(p=2)
                if norm >= boundary:
                    new.mul_((boundary - 1e-8) / norm)

            losses.append(loss.item())

            if verbose and step % 25 == 0:
                print(
                    f"step {step:4d} | "
                    f"loss={loss.item():.3e} | "
                    f"global={loss_global.item():.3e} | "
                    f"local={loss_local.item():.3e} | "
                    f"||x||={norm.item():.3f}"
                )

        return new.detach().cpu().numpy().reshape(self.dim,), losses
    def _squared_norm(self, x, dim=-1, keepdim=True):
        return (x * x).sum(dim=dim, keepdim=keepdim)

    def _mobius_add(self, x, y, eps_small=1e-8):
        x2 = self._squared_norm(x, keepdim=True)
        y2 = self._squared_norm(y, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        den = 1 + 2 * xy + x2 * y2
        return num / (den + eps_small)

    def _mobius_neg(self, x):
        return -x

    def _artanh(self, x):
        x = x.clamp(min=-1 + 1e-6, max=1 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _lambda_x(self, x):
        norm2 = self._squared_norm(x, keepdim=True)
        return 2.0 / (1.0 - norm2 + 1e-8)

    def _log_map(self, x, y):
        # x: (1, dim) ; y: (k, dim)
        u = self._mobius_add(self._mobius_neg(x), y)
        norm_u = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lam = self._lambda_x(x)
        coef = (2.0 / lam) * self._artanh(norm_u) / (norm_u + 1e-8)
        return coef * u

    def _exp_map(self, x, v):
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lam = self._lambda_x(x)
        second = torch.tanh(lam * norm_v / 2.0) * (v / (norm_v + 1e-8))
        return self._mobius_add(x, second)

    def _project_to_ball(self, x, max_norm=1 - 1e-6):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        mask = norm >= max_norm
        if mask.any():
            x = x / norm * max_norm
        return x

    def hyperbolic_barycenter(self, points, weights=None, n_steps=100, tol=1e-6, alpha=1.0, device=None):
        """
        Compute weighted Fréchet mean in the Poincaré ball using log/exp maps.
        points: (k, dim) tensor, weights: (k,) tensor or None.
        Returns x of shape (1, dim)
        """
        if device is None:
            device = points.device
        points = points.to(device)
        k, dim = points.shape
        if weights is None:
            weights = torch.ones(k, device=device) / float(k)
        else:
            weights = weights.to(device).float()
            if weights.sum() <= 0:
                weights = torch.ones_like(weights) / float(k)
            else:
                weights = weights / weights.sum()

        # init: weighted euclidean mean projected into ball
        x = (weights.view(-1, 1) * points).sum(dim=0, keepdim=True)
        x = self._project_to_ball(x)

        for i in range(n_steps):
            v = self._log_map(x, points)  # (k, dim)
            v_bar = (weights.view(-1, 1) * v).sum(dim=0, keepdim=True)
            norm_vbar = v_bar.norm()
            if norm_vbar < tol:
                break
            x_new = self._exp_map(x, alpha * v_bar)
            x_new = self._project_to_ball(x_new)
            x = x_new

        return x

