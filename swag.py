from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


class MultiPrototypes(nn.Module):
    """Linear prototype scorer on the unit sphere."""

    def __init__(self, input_dim: int, num_prototypes: Union[int, Sequence[int]]) -> None:
        super().__init__()
        if isinstance(num_prototypes, int):
            num_prototypes = [num_prototypes]
        self.num_prototypes = [int(k) for k in num_prototypes]
        if any(k <= 0 for k in self.num_prototypes):
            raise ValueError(f"num_prototypes must be positive, got {self.num_prototypes}")

        self.prototypes = nn.ModuleList(
            [nn.Linear(input_dim, k, bias=False) for k in self.num_prototypes]
        )
        for layer in self.prototypes:
            nn.init.orthogonal_(layer.weight, gain=1.0)

    def get_normalized_weights(self) -> List[Tensor]:
        return [F.normalize(layer.weight, dim=1, p=2) for layer in self.prototypes]

    def ortholoss(self) -> Tensor:
        loss: Optional[Tensor] = None
        for layer in self.prototypes:
            w = F.normalize(layer.weight, dim=1, p=2)
            gram = w @ w.t()
            eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            term = F.mse_loss(gram, eye)
            loss = term if loss is None else loss + term
        assert loss is not None
        return loss / len(self.prototypes)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = F.normalize(x, dim=1, p=2)
        outputs: List[Tensor] = []
        for layer in self.prototypes:
            w = F.normalize(layer.weight, dim=1, p=2)
            outputs.append(F.linear(x, w))
        return outputs


class NTXentLoss(nn.Module):
    """Node-wise InfoNCE/NT-Xent ablation."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"NTXentLoss expects equal shapes, got {z1.shape} and {z2.shape}")
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        logits = (z1 @ z2.t()) / max(self.temperature, 1e-12)
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class SwAGLoss(nn.Module):

    def __init__(
        self,
        temp: float = 0.1,
        eps: float = 0.05,
        sk_iter: int = 3,
        use_sinkhorn: bool = True,
        asymmetric_mode: str = "symmetric",
        swag_reverse_weight: Optional[float] = None,
        use_markov_stability: bool = False,
        markov_weight: float = 0.0,
        markov_times: Optional[Sequence[int]] = None,
        markov_adj_norm: str = "row",
        markov_self_loops: bool = True,
        markov_fast_cumulative: bool = True,
        markov_target_only: bool = False,
        markov_time_sampling: str = "none",
        markov_update_interval: int = 1,
        markov_interval_rescale: bool = True,
    ) -> None:
        super().__init__()
        self.temp = float(temp)
        self.eps = float(eps)
        self.sk_iter = int(sk_iter)
        self.use_sinkhorn = bool(use_sinkhorn)
        self.asymmetric_mode = str(asymmetric_mode)
        self.swag_reverse_weight = swag_reverse_weight

        self.use_markov_stability = bool(use_markov_stability)
        self.markov_weight = float(markov_weight)
        self.markov_times = [int(t) for t in (markov_times if markov_times is not None else [1, 2, 4])]
        self.markov_adj_norm = str(markov_adj_norm)
        self.markov_self_loops = bool(markov_self_loops)
        self.markov_fast_cumulative = bool(markov_fast_cumulative)
        self.markov_target_only = bool(markov_target_only)
        self.markov_time_sampling = str(markov_time_sampling)
        self.markov_update_interval = max(1, int(markov_update_interval))
        self.markov_interval_rescale = bool(markov_interval_rescale)
        self.current_epoch = 0
        self.last_logs: Dict[str, Union[Tensor, float]] = {}

        if self.asymmetric_mode not in {"symmetric", "v1_predicts_v2", "v2_predicts_v1"}:
            raise ValueError(f"Unknown asymmetric_mode={asymmetric_mode}")
        if self.markov_adj_norm not in {"row", "sym", "none"}:
            raise ValueError("markov_adj_norm must be one of {'row', 'sym', 'none'}")
        if self.markov_time_sampling not in {"none", "cyclic", "random"}:
            raise ValueError("markov_time_sampling must be one of {'none', 'cyclic', 'random'}")

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @torch.no_grad()
    def sinkhorn_knopp(self, scores: Tensor) -> Tensor:
        if scores.dim() != 2:
            raise ValueError(f"Sinkhorn expects [B,K] scores, got {scores.shape}")
        B, K = scores.shape
        device, dtype = scores.device, scores.dtype

        log_q = (scores / max(self.eps, 1e-12)).t()  # K x B
        log_q = log_q - torch.logsumexp(log_q.reshape(-1), dim=0)

        log_mu = -torch.log(torch.tensor(float(K), device=device, dtype=dtype)).view(1, 1)
        log_nu = -torch.log(torch.tensor(float(B), device=device, dtype=dtype)).view(1, 1)

        for _ in range(max(1, self.sk_iter)):
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) + log_mu
            log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) + log_nu

        return torch.exp(log_q).t() * B

    @torch.no_grad()
    def target_assignment(self, scores: Tensor) -> Tensor:
        if self.use_sinkhorn:
            return self.sinkhorn_knopp(scores)
        return F.softmax(scores / max(self.eps, 1e-12), dim=1)

    def _swapped_ce(self, sc_1: Tensor, sc_2: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
        log_p1 = F.log_softmax(sc_1 / max(self.temp, 1e-12), dim=1)
        log_p2 = F.log_softmax(sc_2 / max(self.temp, 1e-12), dim=1)

        ce_1_to_2 = -(q2 * log_p1).sum(dim=1).mean()  # view 1 predicts view 2 target
        ce_2_to_1 = -(q1 * log_p2).sum(dim=1).mean()  # view 2 predicts view 1 target

        if self.swag_reverse_weight is not None:
            return ce_1_to_2 + float(self.swag_reverse_weight) * ce_2_to_1
        if self.asymmetric_mode == "v1_predicts_v2":
            return ce_1_to_2
        if self.asymmetric_mode == "v2_predicts_v1":
            return ce_2_to_1
        return ce_1_to_2 + ce_2_to_1

    def _add_self_loops_sparse(self, A: Tensor) -> Tensor:
        A = A.coalesce()
        n = A.size(0)
        idx = torch.arange(n, device=A.device)
        eye_idx = torch.stack([idx, idx], dim=0)
        eye_val = torch.ones(n, device=A.device, dtype=A.dtype)
        indices = torch.cat([A.indices(), eye_idx], dim=1)
        values = torch.cat([A.values(), eye_val], dim=0)
        return torch.sparse_coo_tensor(indices, values, A.shape, device=A.device, dtype=A.dtype).coalesce()

    def _normalize_adj(self, A: Optional[Tensor], like: Tensor) -> Optional[Tensor]:
        if A is None:
            return None
        A = A.to(device=like.device, dtype=like.dtype)
        if not A.is_sparse:
            if self.markov_self_loops:
                A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            if self.markov_adj_norm == "none":
                return A
            deg = A.sum(dim=1).clamp_min(1e-12)
            if self.markov_adj_norm == "row":
                return A / deg.unsqueeze(1)
            if self.markov_adj_norm == "sym":
                d_inv_sqrt = deg.rsqrt()
                return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
            raise ValueError(f"Unknown adjacency normalization {self.markov_adj_norm}")

        A = A.coalesce()
        if self.markov_self_loops:
            A = self._add_self_loops_sparse(A)
        if self.markov_adj_norm == "none":
            return A

        row, col = A.indices()
        val = A.values()
        deg = torch.zeros(A.size(0), device=A.device, dtype=A.dtype)
        deg.scatter_add_(0, row, val)
        deg = deg.clamp_min(1e-12)
        if self.markov_adj_norm == "row":
            new_val = val / deg[row]
        elif self.markov_adj_norm == "sym":
            new_val = val / torch.sqrt(deg[row] * deg[col]).clamp_min(1e-12)
        else:
            raise ValueError(f"Unknown adjacency normalization {self.markov_adj_norm}")
        return torch.sparse_coo_tensor(A.indices(), new_val, A.shape, device=A.device, dtype=A.dtype).coalesce()

    @staticmethod
    def _adj_mm(A: Optional[Tensor], X: Tensor) -> Tensor:
        if A is None:
            return X
        if A.is_sparse:
            return torch.sparse.mm(A, X)
        return A @ X

    def _propagate(self, A: Optional[Tensor], X: Tensor, steps: int) -> Tensor:
        out = X
        for _ in range(max(0, int(steps))):
            out = self._adj_mm(A, out)
        return out

    def _propagate_to_times_cumulative(self, A: Optional[Tensor], X: Tensor, times: Sequence[int]) -> Dict[int, Tensor]:
        unique_times = sorted({int(t) for t in times if int(t) > 0})
        if not unique_times:
            return {}
        wanted = set(unique_times)
        out = X
        result: Dict[int, Tensor] = {}
        for step in range(1, unique_times[-1] + 1):
            out = self._adj_mm(A, out)
            if step in wanted:
                result[step] = out
        return result

    def _valid_markov_times(self) -> List[int]:
        return sorted({int(t) for t in self.markov_times if int(t) > 0})

    def _should_run_markov(self) -> bool:
        return (
            self.use_markov_stability
            and self.markov_weight > 0.0
            and (self.current_epoch % self.markov_update_interval) == 0
        )

    def _select_markov_times(self, valid_times: Sequence[int]) -> List[int]:
        times = list(valid_times)
        if not times:
            return []
        if self.markov_time_sampling == "none":
            return times
        if self.markov_time_sampling == "cyclic":
            idx = (self.current_epoch // self.markov_update_interval) % len(times)
            return [times[idx]]
        if self.markov_time_sampling == "random":
            idx = torch.randint(0, len(times), (1,)).item()
            return [times[idx]]
        raise RuntimeError(f"Unhandled markov_time_sampling={self.markov_time_sampling}")

    @staticmethod
    def _entropy(Q: Tensor) -> Tensor:
        return -(Q * torch.log(Q.clamp_min(1e-12))).sum(dim=1).mean()

    @staticmethod
    def _active_count(Q: Tensor) -> Tensor:
        return (Q.argmax(dim=1).bincount(minlength=Q.size(1)) > 0).float().sum()

    def forward(
        self,
        proto_scores_1: List[Tensor],
        proto_scores_2: List[Tensor],
        adj: Optional[Tensor] = None,
        return_parts: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Union[Tensor, float]]]]:
        if len(proto_scores_1) != len(proto_scores_2):
            raise ValueError("Both views must have the same number of prototype heads.")
        if not proto_scores_1:
            raise ValueError("At least one prototype head is required.")

        base_loss: Union[float, Tensor] = 0.0
        markov_loss: Union[float, Tensor] = 0.0
        logs: Dict[str, Union[Tensor, float]] = {}

        markov_active = self._should_run_markov()
        valid_times = self._valid_markov_times() if markov_active else []
        selected_times = self._select_markov_times(valid_times) if markov_active else []
        A_markov = self._normalize_adj(adj, proto_scores_1[0]) if markov_active else None

        for head, (sc1, sc2) in enumerate(zip(proto_scores_1, proto_scores_2)):
            with torch.no_grad():
                q1 = self.target_assignment(sc1.detach())
                q2 = self.target_assignment(sc2.detach())

            head_base = self._swapped_ce(sc1, sc2, q1, q2)
            base_loss = base_loss + head_base

            if markov_active and selected_times:
                if self.markov_fast_cumulative:
                    with torch.no_grad():
                        target_1 = self._propagate_to_times_cumulative(A_markov, sc1.detach(), selected_times)
                        target_2 = self._propagate_to_times_cumulative(A_markov, sc2.detach(), selected_times)
                    if self.markov_target_only:
                        pred_1 = {int(t): sc1 for t in selected_times}
                        pred_2 = {int(t): sc2 for t in selected_times}
                    else:
                        pred_1 = self._propagate_to_times_cumulative(A_markov, sc1, selected_times)
                        pred_2 = self._propagate_to_times_cumulative(A_markov, sc2, selected_times)
                else:
                    target_1, target_2, pred_1, pred_2 = {}, {}, {}, {}
                    for t in selected_times:
                        t = int(t)
                        with torch.no_grad():
                            target_1[t] = self._propagate(A_markov, sc1.detach(), t)
                            target_2[t] = self._propagate(A_markov, sc2.detach(), t)
                        pred_1[t] = sc1 if self.markov_target_only else self._propagate(A_markov, sc1, t)
                        pred_2[t] = sc2 if self.markov_target_only else self._propagate(A_markov, sc2, t)

                head_markov: Union[float, Tensor] = 0.0
                for t in selected_times:
                    t = int(t)
                    with torch.no_grad():
                        q1_t = self.target_assignment(target_1[t])
                        q2_t = self.target_assignment(target_2[t])
                    head_markov = head_markov + self._swapped_ce(pred_1[t], pred_2[t], q1_t, q2_t)
                head_markov = head_markov / len(selected_times)
                if self.markov_update_interval > 1 and self.markov_interval_rescale:
                    head_markov = head_markov * float(self.markov_update_interval)
                markov_loss = markov_loss + head_markov

            if head == 0:
                with torch.no_grad():
                    logs["target_entropy_v1"] = self._entropy(q1).detach()
                    logs["target_entropy_v2"] = self._entropy(q2).detach()
                    logs["target_active_v1"] = self._active_count(q1).detach()
                    logs["target_active_v2"] = self._active_count(q2).detach()
                    logs["base_loss_head0"] = head_base.detach()

        num_heads = len(proto_scores_1)
        assert isinstance(base_loss, Tensor)
        base_loss = base_loss / num_heads
        if isinstance(markov_loss, Tensor):
            markov_loss = markov_loss / num_heads
        else:
            markov_loss = base_loss.new_tensor(0.0)

        total = base_loss + self.markov_weight * markov_loss
        self.last_logs = {
            "swag_base_loss": base_loss.detach(),
            "swag_markov_loss": markov_loss.detach(),
            "swag_markov_weighted": (self.markov_weight * markov_loss).detach(),
            "swag_total_loss": total.detach(),
            "swag_markov_on": float(self.use_markov_stability),
            "swag_markov_active": float(markov_active),
            "swag_markov_weight": float(self.markov_weight),
            "swag_markov_fast_cumulative": float(self.markov_fast_cumulative),
            "swag_markov_target_only": float(self.markov_target_only),
            "swag_markov_num_all_times": float(len(valid_times)),
            "swag_markov_num_selected_times": float(len(selected_times)),
            "swag_markov_selected_time": float(selected_times[0]) if len(selected_times) == 1 else -1.0,
        }
        self.last_logs.update(logs)

        if return_parts:
            return total, self.last_logs
        return total


class GraphSwAG(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        proj_dim: int = 128,
        proj_hidden_dim: int = 512,
        num_prototypes: Union[int, Sequence[int]] = 64,
        temp: float = 0.1,
        eps: float = 0.05,
        sk_iter: int = 3,
        use_proj: bool = True,
        loss_type: str = "swag",
        use_sinkhorn: bool = True,
        asymmetric_mode: str = "symmetric",
        swag_reverse_weight: Optional[float] = None,
        use_markov_stability: bool = False,
        markov_weight: float = 0.0,
        markov_times: Optional[Sequence[int]] = None,
        markov_adj_norm: str = "row",
        markov_self_loops: bool = True,
        markov_fast_cumulative: bool = True,
        markov_target_only: bool = False,
        markov_time_sampling: str = "none",
        markov_update_interval: int = 1,
        markov_interval_rescale: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_projection = bool(use_proj)
        self.loss_type = str(loss_type)

        if self.use_projection:
            self.projection_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(encoder_output_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.GELU(),
                nn.Linear(proj_hidden_dim, proj_dim),
            )
            proto_dim = proj_dim
        else:
            self.projection_head = None
            proto_dim = encoder_output_dim

        if self.loss_type == "swag":
            self.prototypes: Optional[MultiPrototypes] = MultiPrototypes(proto_dim, num_prototypes)
            self.criterion: nn.Module = SwAGLoss(
                temp=temp,
                eps=eps,
                sk_iter=sk_iter,
                use_sinkhorn=use_sinkhorn,
                asymmetric_mode=asymmetric_mode,
                swag_reverse_weight=swag_reverse_weight,
                use_markov_stability=use_markov_stability,
                markov_weight=markov_weight,
                markov_times=markov_times,
                markov_adj_norm=markov_adj_norm,
                markov_self_loops=markov_self_loops,
                markov_fast_cumulative=markov_fast_cumulative,
                markov_target_only=markov_target_only,
                markov_time_sampling=markov_time_sampling,
                markov_update_interval=markov_update_interval,
                markov_interval_rescale=markov_interval_rescale,
            )
        elif self.loss_type == "ntxent":
            self.prototypes = None
            self.criterion = NTXentLoss(temperature=temp)
        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

    def forward_encoder(self, x: Tensor, **kwargs) -> Union[Tensor, dict, tuple]:
        return self.encoder(x, **kwargs)

    def forward_projection(self, embeddings: Tensor, return_raw: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        z_raw = self.projection_head(embeddings) if self.projection_head is not None else embeddings
        z = F.normalize(z_raw, dim=1, p=2)
        return (z, z_raw) if return_raw else z

    def prototype_scores(self, z: Tensor) -> List[Tensor]:
        if self.prototypes is None:
            raise RuntimeError("prototype_scores called when loss_type != 'swag'.")
        return self.prototypes(z)

    def compute_loss(
        self,
        view1: Tensor,
        view2: Tensor,
        adj: Optional[Tensor] = None,
        epoch: Optional[int] = None,
        return_parts: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Union[Tensor, float]]]]:
        z1 = self.forward_projection(view1)
        z2 = self.forward_projection(view2)

        if self.loss_type == "ntxent":
            loss = self.criterion(z1, z2)
            if return_parts:
                return loss, {"ntxent_loss": loss.detach(), "total_loss": loss.detach()}
            return loss

        if epoch is not None and hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(int(epoch))
        scores1 = self.prototype_scores(z1)
        scores2 = self.prototype_scores(z2)
        total, parts = self.criterion(scores1, scores2, adj=adj, return_parts=True)
        if return_parts:
            parts = dict(parts)
            parts["total_loss"] = total.detach()
            return total, parts
        return total

    def get_embeddings(self, x: Tensor, **encoder_kwargs) -> Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward_encoder(x, **encoder_kwargs)
            if isinstance(out, dict):
                emb = out.get("z")
            elif isinstance(out, tuple):
                emb = out[0]
            else:
                emb = out
            if emb is None:
                raise ValueError("Encoder did not return embeddings.")
            return emb.detach()
