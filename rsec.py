from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSECConfig:
    channels: int = 64
    dropout: float = 0.1
    dilations: str = "1,2,4,8,16"
    gamma_init: float = 1e-3
    num_heads: int = 4
    ffn_ratio: int = 4
    decoder_layers: int = 1
    minimum_background_gap: int = 1
    max_decode_events: int = 8

    def __post_init__(self) -> None:
        if self.channels % self.num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if self.ffn_ratio < 1 or self.decoder_layers < 1:
            raise ValueError("ffn_ratio and decoder_layers must be positive")
        if self.minimum_background_gap != 1 or self.max_decode_events < 1:
            raise ValueError("RSEC requires a one-base gap and a positive decode cap")
        if any(int(value) <= 0 for value in self.dilations.split(",")):
            raise ValueError("dilations must be positive integers")


def init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        result = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return result if mask is None else result * mask[:, None].to(result.dtype)


class MaskSafeHCEStage(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.n1 = ChannelLayerNorm1d(channels)
        self.c1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.n2 = ChannelLayerNorm1d(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask[:, None].to(x.dtype)
        update = self.c1(self.n1(x * mask_float, mask)) * mask_float
        update = self.drop(F.gelu(update)) * mask_float
        return self.n2(self.c2(update) * mask_float, mask) * mask_float


class MaskSafeHCETokenEncoder(nn.Module):
    def __init__(self, channels: int, dropout: float, dilations: Sequence[int], gamma: float):
        super().__init__()
        self.stem = nn.Conv1d(4, channels, 3, padding=1)
        self.norm = ChannelLayerNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.stages = nn.ModuleList(
            [MaskSafeHCEStage(channels, dilation, dropout) for dilation in dilations]
        )
        self.g = nn.ParameterList(
            [nn.Parameter(torch.full((1, channels, 1), gamma)) for _ in dilations]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask[:, None].to(x.dtype)
        hidden = self.drop(F.gelu(self.norm(self.stem(x * mask_float) * mask_float, mask)))
        hidden = hidden * mask_float
        for stage, gamma in zip(self.stages, self.g):
            hidden = (hidden + gamma * stage(hidden, mask)) * mask_float
        return hidden


class HCEFeatureEncoder(nn.Module):
    def __init__(self, cfg: RSECConfig):
        super().__init__()
        dilations = [int(value) for value in cfg.dilations.split(",")]
        self.token_encoder = MaskSafeHCETokenEncoder(
            cfg.channels, cfg.dropout, dilations, cfg.gamma_init
        )
        self.apply(init_module)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        encoded = self.token_encoder(x * mask[:, None].to(x.dtype), mask).transpose(1, 2)
        return encoded * mask[..., None]


class GlobalQueryCrossAttentionBlock(nn.Module):
    def __init__(self, cfg: RSECConfig):
        super().__init__()
        channels = cfg.channels
        self.query_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * cfg.ffn_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(channels * cfg.ffn_ratio, channels),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.query_norm(queries)
        update, _ = self.cross_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=~sequence_mask.bool(),
            need_weights=False,
        )
        states = queries + update
        return states + self.ffn(self.ffn_norm(states))


def sinusoidal_position_encoding(
    length: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    even_channels = torch.arange(0, channels, 2, device=device, dtype=torch.float32)
    angle = position * torch.exp(-math.log(10000.0) * even_channels / channels)[None]
    result = torch.zeros(length, channels, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(angle)
    if channels > 1:
        result[:, 1::2] = torch.cos(angle[:, : result[:, 1::2].shape[1]])
    return result[None].to(dtype)


def finite_padding_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def build_interval_used_mask(
    history_starts: torch.Tensor,
    history_ends: torch.Tensor,
    history_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    minimum_background_gap: int,
) -> torch.Tensor:
    if history_starts.shape != history_ends.shape or history_mask.shape != history_starts.shape:
        raise ValueError("invalid history tensors")
    length = sequence_mask.size(-1)
    positions = torch.arange(length, device=history_starts.device)
    lead = history_starts.shape[:-1]
    batch = lead[0]
    sequence = sequence_mask.reshape((batch,) + (1,) * (len(lead) - 1) + (length,))
    sequence = sequence.expand(*lead, length)
    position_shape = (1,) * len(lead) + (length,)
    used = (
        (positions.reshape(position_shape) >= history_starts[..., None] - minimum_background_gap)
        & (positions.reshape(position_shape) <= history_ends[..., None] + minimum_background_gap)
        & history_mask[..., None]
    ).any(-2)
    return used & sequence


def build_dynamic_free_components(
    history_starts: torch.Tensor,
    history_ends: torch.Tensor,
    history_mask: torch.Tensor,
    sequence_lengths: torch.Tensor,
    minimum_background_gap: int,
) -> dict[str, torch.Tensor]:
    lead = history_starts.shape[:-1]
    capacity = history_starts.size(-1)
    lengths = torch.broadcast_to(sequence_lengths, lead)
    large = lengths[..., None] + 1
    order = torch.argsort(
        torch.where(history_mask, history_starts, large), dim=-1, stable=True
    )
    starts = history_starts.gather(-1, order)
    ends = history_ends.gather(-1, order)
    count = history_mask.sum(-1).long()
    left_values: list[torch.Tensor] = []
    right_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    gap = minimum_background_gap + 1
    for component in range(capacity + 1):
        left = torch.zeros_like(lengths) if component == 0 else ends[..., component - 1] + gap
        event_right = (
            starts[..., min(component, capacity - 1)] - gap
            if capacity
            else torch.zeros_like(lengths)
        )
        right = torch.where(count == component, lengths - 1, event_right)
        valid = (
            (component <= count)
            & (left <= right)
            & (left >= 0)
            & (right < lengths)
        )
        left_values.append(left)
        right_values.append(right)
        valid_values.append(valid)
    valid_tensor = torch.stack(valid_values, -1)
    return {
        "component_left": torch.stack(left_values, -1),
        "component_right": torch.stack(right_values, -1),
        "component_valid": valid_tensor,
        "component_count": valid_tensor.sum(-1),
    }


def legal_interval_distribution(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    component_left: torch.Tensor,
    component_right: torch.Tensor,
    component_valid: torch.Tensor,
    sequence_mask: torch.Tensor,
    return_occupancy: bool,
) -> dict[str, torch.Tensor]:
    squeeze = start_logits.ndim == 2
    if squeeze:
        start_logits = start_logits[:, None]
        end_logits = end_logits[:, None]
        component_left = component_left[:, None]
        component_right = component_right[:, None]
        component_valid = component_valid[:, None]
    if start_logits.shape != end_logits.shape or start_logits.ndim != 3:
        raise ValueError("boundary logits must have shape [B,Q,L]")

    batch, states, length = start_logits.shape
    positions = torch.arange(length, device=start_logits.device)[None, None]
    starts = start_logits.float()
    ends = end_logits.float()
    log_partition = torch.full((batch, states), -torch.inf, device=starts.device)
    best_score = torch.full_like(log_partition, -torch.inf)
    best_start = torch.zeros(batch, states, dtype=torch.long, device=starts.device)
    best_end = torch.zeros_like(best_start)
    occupancy = torch.zeros(batch, states, length, device=starts.device)
    start_marginal = torch.zeros_like(occupancy)
    end_marginal = torch.zeros_like(occupancy)
    legal_span_count = torch.zeros(batch, states, dtype=torch.long, device=starts.device)
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for component in range(component_left.size(-1)):
        inside = (
            (positions >= component_left[..., component, None])
            & (positions <= component_right[..., component, None])
            & component_valid[..., component, None]
            & sequence_mask[:, None]
        )
        component_length = (
            component_right[..., component] - component_left[..., component] + 1
        ).clamp_min(0)
        legal_span_count += torch.where(
            component_valid[..., component],
            component_length * (component_length + 1) // 2,
            torch.zeros_like(component_length),
        )
        masked_start = starts.masked_fill(~inside, -torch.inf)
        masked_end = ends.masked_fill(~inside, -torch.inf)
        suffix = torch.logcumsumexp(torch.flip(masked_end, (-1,)), -1).flip(-1)
        component_partition = torch.logsumexp(masked_start + suffix, -1)
        log_partition = torch.logaddexp(log_partition, component_partition)

        suffix_max, suffix_index = torch.flip(masked_end, (-1,)).cummax(-1)
        suffix_max = torch.flip(suffix_max, (-1,))
        suffix_index = length - 1 - torch.flip(suffix_index, (-1,))
        candidate = masked_start + suffix_max
        candidate_start = candidate.argmax(-1)
        candidate_end = suffix_index.gather(-1, candidate_start[..., None]).squeeze(-1)
        candidate_score = candidate.gather(-1, candidate_start[..., None]).squeeze(-1)
        take = candidate_score > best_score
        best_score = torch.where(take, candidate_score, best_score)
        best_start = torch.where(take, candidate_start, best_start)
        best_end = torch.where(take, candidate_end, best_end)
        cache.append((inside, masked_start, masked_end, suffix))

    has_legal_span = torch.isfinite(log_partition)
    best_log_probability = torch.where(
        has_legal_span,
        best_score - log_partition,
        torch.full_like(best_score, -torch.inf),
    )
    for inside, masked_start, masked_end, suffix in cache:
        prefix = torch.logcumsumexp(masked_start, -1)
        log_occupancy = prefix + suffix - log_partition[..., None]
        occupancy += torch.where(
            inside & has_legal_span[..., None],
            log_occupancy.exp(),
            torch.zeros_like(log_occupancy),
        )
        start_marginal += torch.where(
            inside & has_legal_span[..., None],
            (masked_start + suffix - log_partition[..., None]).exp(),
            torch.zeros_like(masked_start),
        )
        end_marginal += torch.where(
            inside & has_legal_span[..., None],
            (masked_end + prefix - log_partition[..., None]).exp(),
            torch.zeros_like(masked_end),
        )

    result = {
        "span_log_partition": log_partition,
        "has_legal_span": has_legal_span,
        "best_start": best_start,
        "best_end": best_end,
        "best_span_raw_score": best_score,
        "best_conditional_probability": best_log_probability.exp(),
        "legal_span_count": legal_span_count,
        "occupancy_probabilities": (
            occupancy.clamp(0, 1) if return_occupancy else torch.empty(0, device=starts.device)
        ),
    }
    return {
        key: value[:, 0] if squeeze and value.ndim >= 2 and value.shape[1] == 1 else value
        for key, value in result.items()
    }


def dynamic_subset_incidence(
    event_capacity: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if event_capacity > 20:
        raise RuntimeError("exact subset training is not tractable above 20 events")
    states = 1 << event_capacity
    if event_capacity == 0:
        incidence = torch.empty(1, 0, dtype=torch.bool, device=device)
    else:
        masks = torch.arange(states, device=device, dtype=torch.long)[:, None]
        bits = torch.arange(event_capacity, device=device, dtype=torch.long)[None]
        incidence = ((masks >> bits) & 1).bool()
    return incidence, incidence.sum(-1).long()


def build_subset_training_states(
    gold_starts: torch.Tensor,
    gold_ends: torch.Tensor,
    gold_event_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch, capacity = gold_starts.shape
    incidence, popcount = dynamic_subset_incidence(capacity, gold_starts.device)
    states = incidence.size(0)
    expanded = incidence[None].expand(batch, -1, -1)
    subset_valid = ~(expanded & ~gold_event_mask[:, None]).any(-1)
    history_mask = expanded & gold_event_mask[:, None]
    history_starts = gold_starts[:, None].expand(-1, states, -1)
    history_ends = gold_ends[:, None].expand(-1, states, -1)
    lengths = sequence_mask.sum(-1).long()
    used = build_interval_used_mask(
        history_starts, history_ends, history_mask, sequence_mask, 1
    )
    components = build_dynamic_free_components(
        history_starts, history_ends, history_mask, lengths[:, None], 1
    )
    return {
        "subset_incidence": incidence,
        "subset_popcount": popcount,
        "subset_valid": subset_valid,
        "subset_used_mask": used,
        **components,
    }


def canonicalize_gold_intervals(
    starts: torch.Tensor, ends: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    sentinel = torch.iinfo(starts.dtype).max
    end_order = torch.argsort(torch.where(mask, ends, sentinel), dim=-1, stable=True)
    starts = starts.gather(-1, end_order)
    ends = ends.gather(-1, end_order)
    mask = mask.gather(-1, end_order)
    start_order = torch.argsort(torch.where(mask, starts, sentinel), dim=-1, stable=True)
    starts = starts.gather(-1, start_order)
    ends = ends.gather(-1, start_order)
    mask = mask.gather(-1, start_order)
    valid_order = torch.argsort((~mask).long(), dim=-1, stable=True)
    return (
        starts.gather(-1, valid_order),
        ends.gather(-1, valid_order),
        mask.gather(-1, valid_order),
    )


class RSEC(nn.Module):

    training_objective_mode = "fixed_left_to_right"

    def __init__(self, cfg: RSECConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = HCEFeatureEncoder(cfg)
        self.memory_norm = nn.LayerNorm(cfg.channels)
        self.used_region_embedding = nn.Parameter(torch.empty(cfg.channels))
        self.event_query = nn.Parameter(torch.empty(cfg.channels))
        self.event_query_norm = nn.LayerNorm(cfg.channels)
        self.event_cross_attention = GlobalQueryCrossAttentionBlock(cfg)
        self.extra_event_cross_attention = nn.ModuleList(
            GlobalQueryCrossAttentionBlock(cfg) for _ in range(cfg.decoder_layers - 1)
        )
        self.null_head = nn.Linear(cfg.channels, 1)
        self.start_q_proj = nn.Linear(cfg.channels, cfg.channels)
        self.start_k_proj = nn.Linear(cfg.channels, cfg.channels)
        self.end_q_proj = nn.Linear(cfg.channels, cfg.channels)
        self.end_k_proj = nn.Linear(cfg.channels, cfg.channels)
        self.apply(init_module)
        nn.init.normal_(self.event_query, std=0.02)
        nn.init.normal_(self.used_region_embedding, std=0.02)

    def encode(
        self, x: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.feature_encoder(x, sequence_mask)
        position = sinusoidal_position_encoding(
            memory.size(1), memory.size(2), memory.device, memory.dtype
        )[0]
        memory = self.memory_norm(memory + position[None])
        memory = memory * sequence_mask[..., None].to(memory.dtype)
        return memory, position

    def decode_event_query(
        self, query: torch.Tensor, memory: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        decoded = self.event_cross_attention(query, memory, sequence_mask)
        for layer in self.extra_event_cross_attention:
            decoded = layer(decoded, memory, sequence_mask)
        return decoded

    def decode_residual_scopes(
        self, memory: torch.Tensor, sequence_mask: torch.Tensor, used_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, states, length = used_mask.shape
        channels = memory.size(-1)
        state_memory = memory[:, None].expand(-1, states, -1, -1)
        state_memory = state_memory + used_mask[..., None].to(memory.dtype) * self.used_region_embedding
        flat_memory = state_memory.reshape(batch * states, length, channels)
        flat_mask = sequence_mask[:, None].expand(-1, states, -1).reshape(batch * states, length)
        query = self.event_query_norm(self.event_query).to(memory.dtype)
        query = query[None, None].expand(batch * states, 1, -1)
        decoded = self.decode_event_query(query, flat_memory, flat_mask)[:, 0]
        decoded = decoded.reshape(batch, states, channels)
        keys = memory[:, None].expand(-1, states, -1, -1)
        scale = math.sqrt(channels)
        start = torch.einsum(
            "bsc,bslc->bsl", self.start_q_proj(decoded), self.start_k_proj(keys)
        ) / scale
        end = torch.einsum(
            "bsc,bslc->bsl", self.end_q_proj(decoded), self.end_k_proj(keys)
        ) / scale
        padding = finite_padding_value(start)
        return {
            "null_logits": self.null_head(decoded).squeeze(-1),
            "start_logits": start.masked_fill(~sequence_mask[:, None], padding),
            "end_logits": end.masked_fill(~sequence_mask[:, None], padding),
        }

    def forward(
        self,
        x: torch.Tensor,
        sequence_mask: torch.Tensor,
        gold_starts: torch.Tensor,
        gold_ends: torch.Tensor,
        gold_event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gold_starts, gold_ends, gold_event_mask = canonicalize_gold_intervals(
            gold_starts, gold_ends, gold_event_mask
        )
        memory, position = self.encode(x, sequence_mask)
        subset = build_subset_training_states(
            gold_starts, gold_ends, gold_event_mask, sequence_mask
        )
        decoded = self.decode_residual_scopes(
            memory, sequence_mask, subset["subset_used_mask"]
        )
        distribution = legal_interval_distribution(
            decoded["start_logits"],
            decoded["end_logits"],
            subset["component_left"],
            subset["component_right"],
            subset["component_valid"],
            sequence_mask,
            False,
        )
        null_logits = decoded["null_logits"].float()
        span_partition = distribution["span_log_partition"].float()
        total_partition = torch.logaddexp(null_logits, span_partition)
        states = decoded["start_logits"].size(1)
        capacity = gold_starts.size(1)
        transition = torch.full(
            (x.size(0), states, capacity), -torch.inf, device=x.device
        )
        for event in range(capacity):
            safe_start = gold_starts[:, event].clamp_min(0)
            safe_end = gold_ends[:, event].clamp_min(0)
            raw_score = decoded["start_logits"].float().gather(
                -1, safe_start[:, None, None].expand(-1, states, 1)
            ).squeeze(-1)
            raw_score += decoded["end_logits"].float().gather(
                -1, safe_end[:, None, None].expand(-1, states, 1)
            ).squeeze(-1)
            within = (
                (safe_start[:, None, None] >= subset["component_left"])
                & (safe_end[:, None, None] <= subset["component_right"])
                & subset["component_valid"]
            ).any(-1)
            valid = (
                subset["subset_valid"]
                & gold_event_mask[:, event, None]
                & ~subset["subset_incidence"][None, :, event]
                & within
            )
            transition[:, :, event] = torch.where(
                valid, raw_score - total_partition, transition[:, :, event]
            )
        return {
            "memory": memory,
            "position_encoding": position,
            **subset,
            **decoded,
            "span_log_partition": span_partition,
            "null_log_probabilities": null_logits - total_partition,
            "transition_log_probabilities": transition,
            "gold_event_counts": gold_event_mask.sum(-1).long(),
        }


def fixed_order_nll(
    outputs: dict[str, torch.Tensor], gold_event_counts: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=outputs["null_logits"].device.type, enabled=False):
        transition = outputs["transition_log_probabilities"].float()
        capacity = transition.size(-1)
        steps = torch.arange(capacity, device=transition.device)
        prefix_subset = (1 << steps) - 1
        selected = transition[:, prefix_subset, steps]
        valid = steps[None] < gold_event_counts.long()[:, None]
        path = torch.where(valid, selected, torch.zeros_like(selected)).sum(-1)
        full_subset = (1 << gold_event_counts.long()) - 1
        terminal = outputs["null_log_probabilities"].float().gather(
            1, full_subset[:, None]
        ).squeeze(1)
        per_sequence = -path - terminal
        loss = per_sequence.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("RSEC fixed-order NLL is non-finite")
    return {"loss": loss, "joint_nll": loss, "per_sequence_joint_nll": per_sequence}


@torch.inference_mode()
def decode_from_memory(
    model: RSEC, memory: torch.Tensor, sequence_mask: torch.Tensor
) -> dict[str, Any]:
    batch, length = sequence_mask.shape
    lengths = sequence_mask.sum(-1).long()
    active = lengths > 0
    events: list[list[tuple[int, int]]] = [[] for _ in range(batch)]
    confidences: list[list[float]] = [[] for _ in range(batch)]
    binding_score = torch.zeros(batch, length, device=memory.device)
    termination = ["geometry_exhausted" if not bool(active[i]) else "" for i in range(batch)]
    geometry_guard = (length + 1) // 2
    decode_guard = min(geometry_guard, model.cfg.max_decode_events)
    step = 0

    while bool(active.any()):
        if step >= decode_guard:
            reason = (
                "max_event_safety_cap"
                if model.cfg.max_decode_events <= geometry_guard
                else "geometry_guard_reached"
            )
            for index in active.nonzero(as_tuple=False).flatten().tolist():
                termination[index] = reason
            break

        capacity = max((len(row) for row in events), default=0)
        history_starts = torch.zeros(batch, capacity, dtype=torch.long, device=memory.device)
        history_ends = torch.zeros_like(history_starts)
        history_mask = torch.zeros(batch, capacity, dtype=torch.bool, device=memory.device)
        for index, row in enumerate(events):
            if row:
                history_starts[index, : len(row)] = torch.tensor(
                    [item[0] for item in row], device=memory.device
                )
                history_ends[index, : len(row)] = torch.tensor(
                    [item[1] for item in row], device=memory.device
                )
                history_mask[index, : len(row)] = True

        used = build_interval_used_mask(
            history_starts,
            history_ends,
            history_mask,
            sequence_mask,
            model.cfg.minimum_background_gap,
        )
        components = build_dynamic_free_components(
            history_starts,
            history_ends,
            history_mask,
            lengths,
            model.cfg.minimum_background_gap,
        )
        decoded = model.decode_residual_scopes(memory, sequence_mask, used[:, None])
        null_logit = decoded["null_logits"][:, 0].float()
        distribution = legal_interval_distribution(
            decoded["start_logits"][:, 0],
            decoded["end_logits"][:, 0],
            components["component_left"],
            components["component_right"],
            components["component_valid"],
            sequence_mask,
            True,
        )
        span_partition = distribution["span_log_partition"]
        total_partition = torch.logaddexp(null_logit, span_partition)
        span_probability = torch.where(
            distribution["has_legal_span"],
            (span_partition - total_partition).exp(),
            torch.zeros_like(null_logit),
        )
        best_joint = torch.where(
            distribution["has_legal_span"],
            (distribution["best_span_raw_score"] - total_partition).exp(),
            torch.zeros_like(null_logit),
        )
        choose_span = (
            active
            & distribution["has_legal_span"]
            & (distribution["best_span_raw_score"] > null_logit)
        )
        binding_score = torch.maximum(
            binding_score,
            torch.where(
                active[:, None],
                span_probability[:, None] * distribution["occupancy_probabilities"],
                torch.zeros_like(binding_score),
            ),
        )
        for index in choose_span.nonzero(as_tuple=False).flatten().tolist():
            events[index].append(
                (
                    int(distribution["best_start"][index]),
                    int(distribution["best_end"][index]),
                )
            )
            confidences[index].append(float(best_joint[index]))
        for index in (active & ~choose_span).nonzero(as_tuple=False).flatten().tolist():
            termination[index] = (
                "null" if bool(distribution["has_legal_span"][index]) else "geometry_exhausted"
            )
        active = choose_span
        step += 1

    sorted_rows: list[list[tuple[int, int]]] = []
    sorted_confidences: list[list[float]] = []
    for row, scores in zip(events, confidences):
        order = sorted(range(len(row)), key=lambda index: row[index])
        sorted_rows.append([row[index] for index in order])
        sorted_confidences.append([scores[index] for index in order])
    return {
        "events": sorted_rows,
        "generation_events": events,
        "confidences": sorted_confidences,
        "generation_confidences": confidences,
        "binding_score": binding_score * sequence_mask,
        "termination_reason": termination,
    }


@torch.inference_mode()
def decode(model: RSEC, x: torch.Tensor, sequence_mask: torch.Tensor) -> dict[str, Any]:
    memory, _ = model.encode(x, sequence_mask)
    return decode_from_memory(model, memory, sequence_mask)
