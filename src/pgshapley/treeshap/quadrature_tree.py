"""O(m_q * L) TreeSHAP solver based on Gauss-Legendre quadrature.

This backend implements the two-pass algorithm described in ``paper.tex``
(section "Tree ensemble value functions and induced product structure",
algorithms "Optimal algorithm: first pass" and "Optimal algorithm: second pass").

For each sample, the method walks the tree once to populate per-node
accumulators ``G_u(x_r)`` and per-edge increments ``Delta F_e(x_r)``, and then
performs a bottom-up aggregation that yields all feature Shapley values in
``O(m_q * L)`` time per tree, where ``L`` is the number of leaves and ``m_q``
is the Gauss-Legendre quadrature order.

For exact Shapley values we pick ``m_q = ceil(D/2)``, where ``D`` is the number
of distinct features on the longest root-to-leaf path; smaller choices trade
accuracy for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import PreparedModel, TreeShapBackend
from .unified import UnifiedEnsemble, UnifiedTree


@dataclass
class _PreparedTreeQT:
    """Per-tree precomputed metadata for the quadrature tree backend."""

    tree: UnifiedTree
    tree_weight: float
    max_path_features: int  # D: longest distinct-feature path length

    # For each non-root node: weight of the edge from its parent.
    edge_weight: np.ndarray     # (n_nodes,) float64 (NaN for root)
    inv_edge_weight: np.ndarray  # (n_nodes,) float64 (NaN for root)

    # Post-order traversal (children before parent) for the bottom-up pass.
    postorder: np.ndarray  # (n_nodes,) int32

    # Gauss-Legendre quadrature on [0, 1].
    quad_x: np.ndarray  # (m_q,)
    quad_w: np.ndarray  # (m_q,)


@dataclass
class PreparedQuadratureTreeModel(PreparedModel):
    trees: List[_PreparedTreeQT]


def _edge_weight(tree: UnifiedTree, parent: int, child: int) -> float:
    pw = float(tree.node_weight[parent])
    cw = float(tree.node_weight[child])
    if pw <= 0.0:
        return 0.5
    w = cw / pw
    return float(np.clip(w, 1e-300, 1.0))


class QuadratureTreeShapBackend(TreeShapBackend):
    """Two-pass O(m_q * L) TreeSHAP using Gauss-Legendre quadrature.

    Parameters
    ----------
    m_q:
        Quadrature order. If ``None`` (default), uses ``max(1, (D+1)//2)``
        per tree, where ``D`` is the longest distinct-feature root-to-leaf
        path. This gives Shapley values that are exact up to floating-point
        precision; smaller ``m_q`` trades accuracy for speed.
    """

    def __init__(self, *, m_q: Optional[int] = None):
        super().__init__()
        self._m_q_user = m_q

    def _m_q_for_tree(self, D: int) -> int:
        if self._m_q_user is not None:
            return int(self._m_q_user)
        return max(1, (int(D) + 1) // 2)

    def prepare(self, ensemble: UnifiedEnsemble) -> PreparedQuadratureTreeModel:
        prepared_trees: List[_PreparedTreeQT] = []
        expected_value = np.zeros((ensemble.n_outputs,), dtype=np.float64)

        for tree, t_weight in zip(ensemble.trees, ensemble.tree_weights):
            n_nodes = int(tree.children_left.shape[0])

            edge_weight = np.full(n_nodes, np.nan, dtype=np.float64)
            inv_edge_weight = np.full(n_nodes, np.nan, dtype=np.float64)

            postorder: List[int] = []

            max_d = 0
            path_feats: set[int] = set()

            def rec(u: int) -> None:
                nonlocal max_d
                l = int(tree.children_left[u])
                r = int(tree.children_right[u])
                if l == -1:
                    if len(path_feats) > max_d:
                        max_d = len(path_feats)
                    postorder.append(u)
                    return

                f = int(tree.feature[u])
                for child in (l, r):
                    w_e = _edge_weight(tree, u, child)
                    edge_weight[child] = w_e
                    inv_edge_weight[child] = 1.0 / w_e
                added = f not in path_feats
                if added:
                    path_feats.add(f)
                rec(l)
                rec(r)
                if added:
                    path_feats.remove(f)
                postorder.append(u)

            rec(0)

            m_q = self._m_q_for_tree(max_d)
            leg_x, leg_w = np.polynomial.legendre.leggauss(m_q)
            qx = (0.5 * (leg_x + 1.0)).astype(np.float64)
            qw = (0.5 * leg_w).astype(np.float64)

            # Expected value contribution for this tree.
            root_w = float(tree.node_weight[0])
            if root_w <= 0.0:
                root_w = 1.0
            for u in postorder:
                if int(tree.children_left[u]) == -1:
                    leaf_w = float(tree.node_weight[u])
                    prob = leaf_w / root_w
                    expected_value += t_weight * prob * tree.values[u].astype(np.float64, copy=False)

            prepared_trees.append(
                _PreparedTreeQT(
                    tree=tree,
                    tree_weight=float(t_weight),
                    max_path_features=int(max_d),
                    edge_weight=edge_weight,
                    inv_edge_weight=inv_edge_weight,
                    postorder=np.asarray(postorder, dtype=np.int32),
                    quad_x=qx,
                    quad_w=qw,
                )
            )

        out = PreparedQuadratureTreeModel(
            ensemble=ensemble,
            expected_value=expected_value,
            trees=prepared_trees,
        )
        self.prepared = out
        return out

    def explain(self, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
        if self.prepared is None:
            raise RuntimeError("Backend is not prepared. Call prepare() first.")

        ensemble = self.prepared.ensemble
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != ensemble.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but model has {ensemble.n_features}."
            )

        n_trees_total = len(self.prepared.trees)
        n_trees = n_trees_total if tree_limit is None else min(n_trees_total, int(tree_limit))

        n_samples = X.shape[0]
        out = np.zeros((n_samples, ensemble.n_features, ensemble.n_outputs), dtype=np.float64)

        for t in range(n_trees):
            pt = self.prepared.trees[t]
            for s in range(n_samples):
                out[s] += pt.tree_weight * self._explain_tree_sample(pt, X[s])

        return out

    def _explain_tree_sample(self, pt: _PreparedTreeQT, x: np.ndarray) -> np.ndarray:
        """Compute Shapley contributions of a single tree for a single sample.

        Returns an ``(n_features, n_outputs)`` array (not yet scaled by
        ``tree_weight``).
        """

        tree = pt.tree
        n_nodes = int(tree.children_left.shape[0])
        n_features = int(tree.n_features)
        n_outputs = int(tree.n_outputs)
        qx = pt.quad_x
        qw = pt.quad_w
        m_q = int(qx.shape[0])

        # Accumulators filled by the first pass.
        G_node = np.empty((n_nodes, m_q), dtype=np.float64)
        Delta_F = np.zeros((n_nodes, m_q), dtype=np.float64)

        # Persistent DFS state: current q / F values per feature.
        # q defaults to 1 for features not yet seen on the current path, which
        # makes a_r(q) = 1 and F = 0.
        current_q = np.ones(n_features, dtype=np.float64)
        current_F = np.zeros((n_features, m_q), dtype=np.float64)

        # Current G vector as we descend the tree.
        G_cur = np.ones(m_q, dtype=np.float64)

        def dfs1(u: int) -> None:
            nonlocal G_cur
            # Snapshot G at this node before descending.
            G_node[u] = G_cur

            l = int(tree.children_left[u])
            r = int(tree.children_right[u])
            if l == -1:
                return

            f = int(tree.feature[u])
            thr = float(tree.threshold[u])

            q_old = float(current_q[f])
            F_old = current_F[f].copy()
            a_old = (1.0 - qx) + qx * q_old  # (m_q,)

            for child, is_left in ((l, True), (r, False)):
                inv_w = float(pt.inv_edge_weight[child])
                w_e = float(pt.edge_weight[child])
                if is_left:
                    sat = 1.0 if x[f] <= thr else 0.0
                else:
                    sat = 1.0 if x[f] > thr else 0.0
                q_new = q_old * inv_w * sat
                a_new = (1.0 - qx) + qx * q_new  # (m_q,)

                F_new = (q_new - 1.0) / a_new
                Delta_F[child] = F_new - F_old

                G_saved = G_cur
                G_cur = G_cur * w_e * (a_new / a_old)
                current_q[f] = q_new
                current_F[f] = F_new

                dfs1(child)

                G_cur = G_saved
                current_q[f] = q_old
                current_F[f] = F_old

        dfs1(0)

        # Second pass: bottom-up accumulation of H and Shapley values.
        # H[u, r, :] = sum over leaves v in the subtree rooted at u of
        #              V_v * G[v, r] (a vector of length n_outputs).
        # For each edge (u -> v) splitting on feature f:
        #     phi[f] += sum_r w_r * H[v, r, :] * Delta_F[v, r]
        H = np.zeros((n_nodes, m_q, n_outputs), dtype=np.float64)
        phi = np.zeros((n_features, n_outputs), dtype=np.float64)

        for u in pt.postorder:
            u = int(u)
            l = int(tree.children_left[u])
            if l == -1:
                V_u = tree.values[u].astype(np.float64, copy=False)
                H[u] = G_node[u][:, None] * V_u[None, :]
                continue

            r = int(tree.children_right[u])
            H[u] = H[l] + H[r]

            f = int(tree.feature[u])
            for child in (l, r):
                # sum_r w_r * H[child, r, :] * Delta_F[child, r]
                contrib = (qw[:, None] * Delta_F[child][:, None] * H[child]).sum(axis=0)
                phi[f] += contrib

        return phi
