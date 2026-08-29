from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


@dataclass
class _Totals:
    layer_evaluations: int = 0
    total_dream_tokens: float = 0.0
    kept_dream_tokens: float = 0.0
    pruned_dream_tokens: float = 0.0
    min_pruned_dream_tokens: float | None = None
    max_pruned_dream_tokens: float | None = None

    def add(self, *, total: float, kept: float) -> None:
        pruned = total - kept
        self.layer_evaluations += 1
        self.total_dream_tokens += total
        self.kept_dream_tokens += kept
        self.pruned_dream_tokens += pruned
        if self.min_pruned_dream_tokens is None:
            self.min_pruned_dream_tokens = pruned
            self.max_pruned_dream_tokens = pruned
        else:
            self.min_pruned_dream_tokens = min(self.min_pruned_dream_tokens, pruned)
            self.max_pruned_dream_tokens = max(self.max_pruned_dream_tokens, pruned)

    def merge(self, other: "_Totals") -> None:
        if other.layer_evaluations == 0:
            return
        self.layer_evaluations += other.layer_evaluations
        self.total_dream_tokens += other.total_dream_tokens
        self.kept_dream_tokens += other.kept_dream_tokens
        self.pruned_dream_tokens += other.pruned_dream_tokens
        if self.min_pruned_dream_tokens is None:
            self.min_pruned_dream_tokens = other.min_pruned_dream_tokens
            self.max_pruned_dream_tokens = other.max_pruned_dream_tokens
        else:
            self.min_pruned_dream_tokens = min(
                self.min_pruned_dream_tokens,
                float(other.min_pruned_dream_tokens),
            )
            self.max_pruned_dream_tokens = max(
                self.max_pruned_dream_tokens,
                float(other.max_pruned_dream_tokens),
            )

    def summary(self) -> dict[str, Any]:
        if self.layer_evaluations <= 0:
            raise RuntimeError("No threshold layer evaluations were collected.")
        count = float(self.layer_evaluations)
        return {
            "layer_evaluations": self.layer_evaluations,
            "average_total_dream_tokens": self.total_dream_tokens / count,
            "average_kept_dream_tokens": self.kept_dream_tokens / count,
            "average_pruned_dream_tokens": self.pruned_dream_tokens / count,
            "prune_ratio": (
                self.pruned_dream_tokens / self.total_dream_tokens
                if self.total_dream_tokens > 0
                else 0.0
            ),
            "min_pruned_dream_tokens": self.min_pruned_dream_tokens,
            "max_pruned_dream_tokens": self.max_pruned_dream_tokens,
            "any_tokens_pruned": self.pruned_dream_tokens > 1.0e-6,
        }


class ThresholdPruningCollector:
    """Aggregate threshold decisions and optionally retain proxy tensors.

    One layer evaluation means one MoT layer at one Action denoising step for
    one policy replan. LIBERO evaluation uses batch size one, while ``k_mean``
    also keeps the aggregation well-defined if that changes in the future.
    """

    _PROXY_SCALAR_FIELDS = (
        "proxy_head_dim",
        "full_k_mean",
        "proxy_k_mean",
        "proxy_k_delta_mean",
        "proxy_teacher_recall",
        "proxy_teacher_precision",
        "proxy_teacher_jaccard",
        "proxy_mask_agreement",
        "proxy_false_negative_mean",
        "proxy_false_positive_mean",
        "proxy_score_mae",
        "proxy_score_rmse",
        "full_dense_dream_mass",
        "full_teacher_kept_mass",
        "proxy_mask_kept_full_mass",
        "proxy_full_mass_retention",
    )
    _PROXY_TENSOR_FIELDS = (
        "proxy_channel_indices",
        "full_score",
        "proxy_score",
        "full_keep_mask",
        "proxy_keep_mask",
        "full_dense_dream_mass_per_sample",
        "full_teacher_kept_mass_per_sample",
        "proxy_mask_kept_full_mass_per_sample",
        "proxy_full_mass_retention_per_sample",
    )

    def __init__(
        self,
        *,
        configured_alpha: float,
        expected_layers: int,
        capture_proxy_tensors: bool = False,
    ):
        self.configured_alpha = float(configured_alpha)
        self.expected_layers = int(expected_layers)
        if self.expected_layers <= 0:
            raise ValueError("expected_layers must be positive.")
        self.episodes: list[dict[str, Any]] = []
        self.layer_records: list[dict[str, Any]] = []
        self.capture_proxy_tensors = bool(capture_proxy_tensors)
        self.proxy_tensor_records: list[dict[str, Any]] = []
        self._episode_index: int | None = None
        self._episode_replans: list[tuple[dict[str, Any], _Totals]] = []
        self._episode_layer_records: list[dict[str, Any]] = []
        self._episode_proxy_tensor_records: list[dict[str, Any]] = []
        self._replan_totals: _Totals | None = None
        self._replan_layer_records: list[dict[str, Any]] = []
        self._replan_proxy_tensor_records: list[dict[str, Any]] = []
        self._replan_steps = 0
        self._expected_denoise_steps: int | None = None

    def begin_episode(self, episode_index: int) -> None:
        if self._episode_index is not None:
            raise RuntimeError("Cannot begin an episode while another episode is active.")
        self._episode_index = int(episode_index)
        self._episode_replans = []
        self._episode_layer_records = []
        self._episode_proxy_tensor_records = []

    def abort_episode(self) -> None:
        self._episode_index = None
        self._episode_replans = []
        self._episode_layer_records = []
        self._episode_proxy_tensor_records = []
        self.abort_replan()

    def begin_replan(self, *, expected_denoise_steps: int) -> None:
        if self._episode_index is None:
            raise RuntimeError("Cannot begin a replan outside an episode.")
        if self._replan_totals is not None:
            raise RuntimeError("Cannot begin a replan while another replan is active.")
        self._replan_totals = _Totals()
        self._replan_layer_records = []
        self._replan_proxy_tensor_records = []
        self._replan_steps = 0
        self._expected_denoise_steps = int(expected_denoise_steps)

    def abort_replan(self) -> None:
        self._replan_totals = None
        self._replan_layer_records = []
        self._replan_proxy_tensor_records = []
        self._replan_steps = 0
        self._expected_denoise_steps = None

    def capture_denoise_step(self, records: Iterable[dict[str, Any]]) -> None:
        if self._replan_totals is None:
            raise RuntimeError("Threshold statistics arrived outside an active replan.")
        records = list(records)
        if len(records) != self.expected_layers:
            raise RuntimeError(
                "Threshold pruning was not observed on every MoT layer: "
                f"expected {self.expected_layers}, got {len(records)}."
            )
        seen_layers = set()
        for record in records:
            layer = int(record["layer"])
            if layer in seen_layers:
                raise RuntimeError(f"Duplicate threshold record for layer {layer}.")
            seen_layers.add(layer)
            alpha = float(record["alpha_current"])
            if abs(alpha - self.configured_alpha) > 1.0e-6:
                raise RuntimeError(
                    f"Runtime threshold alpha {alpha} differs from configured alpha "
                    f"{self.configured_alpha}."
                )
            dream_start, dream_end = (int(value) for value in record["slices"]["dream"])
            total = float(dream_end - dream_start)
            kept = float(record["k_mean"])
            if total <= 0 or kept < -1.0e-6 or kept > total + 1.0e-6:
                raise RuntimeError(
                    f"Invalid threshold counts at layer {layer}: kept={kept}, total={total}."
                )
            self._replan_totals.add(total=total, kept=kept)
            pruned = total - kept
            layer_record = {
                "episode": self._episode_index,
                "replan": len(self._episode_replans),
                "denoise_step": self._replan_steps,
                "layer": layer,
                "alpha": alpha,
                "total_dream_tokens": total,
                "kept_dream_tokens": kept,
                "pruned_dream_tokens": pruned,
                "prune_ratio": pruned / total,
                "k_std": record.get("k_std"),
                "k_min": record.get("k_min"),
                "k_max": record.get("k_max"),
                "reuse_inference_step": record.get("reuse_inference_step"),
                "mask_refreshed": record.get("mask_refreshed"),
            }
            if "proxy_head_dim" in record:
                layer_record.update(
                    {field: record.get(field) for field in self._PROXY_SCALAR_FIELDS}
                )
            self._replan_layer_records.append(layer_record)
            if self.capture_proxy_tensors:
                missing = [field for field in self._PROXY_TENSOR_FIELDS if field not in record]
                if missing:
                    raise RuntimeError(
                        "Proxy tensor capture was requested, but threshold record is missing "
                        f"{missing}. Enable proxy_analysis_enabled and save_detailed_tensors."
                    )
                tensor_record = {
                    "episode": self._episode_index,
                    "replan": len(self._episode_replans),
                    "denoise_step": self._replan_steps,
                    "layer": layer,
                    "alpha": alpha,
                    "proxy_head_dim": record.get("proxy_head_dim"),
                }
                tensor_record.update(
                    {field: record[field] for field in self._PROXY_TENSOR_FIELDS}
                )
                self._replan_proxy_tensor_records.append(tensor_record)
        self._replan_steps += 1

    def finish_replan(self) -> dict[str, Any]:
        if self._replan_totals is None:
            raise RuntimeError("No active replan to finish.")
        if self._expected_denoise_steps != self._replan_steps:
            raise RuntimeError(
                "Threshold statistics were not observed at every Action denoising step: "
                f"expected {self._expected_denoise_steps}, got {self._replan_steps}."
            )
        summary = {
            "replan_index": len(self._episode_replans),
            "denoise_steps": self._replan_steps,
            **self._replan_totals.summary(),
        }
        self._episode_replans.append((summary, self._replan_totals))
        self._episode_layer_records.extend(self._replan_layer_records)
        self._episode_proxy_tensor_records.extend(self._replan_proxy_tensor_records)
        self.abort_replan()
        return summary

    def finish_episode(self, *, success: bool) -> dict[str, Any]:
        if self._episode_index is None:
            raise RuntimeError("No active episode to finish.")
        if self._replan_totals is not None:
            raise RuntimeError("Cannot finish an episode while a replan is active.")
        if not self._episode_replans:
            raise RuntimeError("Episode completed without any thresholded policy replans.")
        totals = _Totals()
        for _, replan_totals in self._episode_replans:
            totals.merge(replan_totals)
        summary = {
            "episode": self._episode_index,
            "success": bool(success),
            "replans": len(self._episode_replans),
            "denoise_steps": sum(item[0]["denoise_steps"] for item in self._episode_replans),
            **totals.summary(),
            "replan_summaries": [item[0] for item in self._episode_replans],
        }
        self.episodes.append(summary)
        for record in self._episode_layer_records:
            record["success"] = bool(success)
        self.layer_records.extend(self._episode_layer_records)
        for record in self._episode_proxy_tensor_records:
            record["success"] = bool(success)
        self.proxy_tensor_records.extend(self._episode_proxy_tensor_records)
        self._episode_index = None
        self._episode_replans = []
        self._episode_layer_records = []
        self._episode_proxy_tensor_records = []
        return summary

    def overall_summary(self) -> dict[str, Any]:
        if self._episode_index is not None or self._replan_totals is not None:
            raise RuntimeError("Cannot summarize while collection is active.")
        if not self.episodes:
            raise RuntimeError("No completed threshold episodes were collected.")
        totals = _Totals()
        for episode in self.episodes:
            episode_totals = _Totals(
                layer_evaluations=int(episode["layer_evaluations"]),
                total_dream_tokens=(
                    float(episode["average_total_dream_tokens"])
                    * int(episode["layer_evaluations"])
                ),
                kept_dream_tokens=(
                    float(episode["average_kept_dream_tokens"])
                    * int(episode["layer_evaluations"])
                ),
                pruned_dream_tokens=(
                    float(episode["average_pruned_dream_tokens"])
                    * int(episode["layer_evaluations"])
                ),
                min_pruned_dream_tokens=float(episode["min_pruned_dream_tokens"]),
                max_pruned_dream_tokens=float(episode["max_pruned_dream_tokens"]),
            )
            totals.merge(episode_totals)
        return {
            "threshold_enabled": True,
            "threshold_executed": True,
            "configured_alpha": self.configured_alpha,
            "expected_layers": self.expected_layers,
            "episodes": len(self.episodes),
            "replans": sum(int(episode["replans"]) for episode in self.episodes),
            "denoise_steps": sum(int(episode["denoise_steps"]) for episode in self.episodes),
            **totals.summary(),
            "episode_summaries": self.episodes,
        }


def instrument_threshold_model(model: Any, collector: ThresholdPruningCollector) -> None:
    """Attach threshold/proxy collection to the cached Action inference path."""
    if getattr(model, "_threshold_pruning_instrumented", False):
        raise RuntimeError("Threshold pruning instrumentation is already installed.")
    mot = getattr(model, "mot", None)
    cfg = getattr(mot, "threshold_config", None)
    if cfg is None:
        raise TypeError("Loaded model has no Action-Dream threshold configuration.")
    if not bool(cfg.enabled):
        raise RuntimeError("Action-Dream threshold is disabled in the loaded evaluation model.")
    runtime_alpha = float(mot.current_alpha())
    if runtime_alpha <= 0.0:
        raise RuntimeError(
            f"Action-Dream threshold alpha is {runtime_alpha}; inference uses the dense fast path."
        )
    if abs(runtime_alpha - collector.configured_alpha) > 1.0e-6:
        raise RuntimeError(
            f"Collector alpha {collector.configured_alpha} does not match runtime alpha {runtime_alpha}."
        )

    # Statistics collection does not change the threshold mask or Action
    # output. Detailed CPU tensors remain opt-in through the collector.
    mot.threshold_config = replace(
        cfg,
        log_statistics=True,
        save_detailed_tensors=collector.capture_proxy_tensors,
    )
    mot._collect_training_statistics = True

    original_predict = model._predict_action_noise_with_cache
    original_infer = model.infer_action

    def tracked_predict(*args, **kwargs):
        output = original_predict(*args, **kwargs)
        collector.capture_denoise_step(mot.last_threshold_statistics)
        return output

    def tracked_infer(*args, **kwargs):
        expected_steps = int(kwargs.get("num_inference_steps", 20))
        collector.begin_replan(expected_denoise_steps=expected_steps)
        try:
            output = original_infer(*args, **kwargs)
            collector.finish_replan()
            return output
        except BaseException:
            collector.abort_replan()
            raise

    model._predict_action_noise_with_cache = tracked_predict
    model.infer_action = tracked_infer
    model._threshold_pruning_instrumented = True
