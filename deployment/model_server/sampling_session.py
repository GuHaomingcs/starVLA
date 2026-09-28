"""Per-connection sampling validation and episode-local DVAC calibration."""

import math
from collections import deque

import numpy as np


class SamplingSession:
    def __init__(self, contract):
        self.contract = contract
        self.history = deque()
        self.window_size = None
        self.session_id = None

    def reset(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("reset requires a non-empty session_id")
        self.history.clear()
        self.window_size = None
        self.session_id = session_id

    def parameters(self, sampling):
        mode = sampling.get("mode", "default")
        if mode not in self.contract["sampling_modes"]:
            raise ValueError(f"Sampling mode {mode!r} is not enabled")
        return self._parameters(mode, sampling)

    def finish(self, result, sampling):
        if sampling.get("mode") == "dvac":
            result = {**result, "dvac": self._calibrate_dvac(dict(result["dvac"]), sampling)}
        return result

    def _parameters(self, mode, sampling):
        """Validate conditions already expressed in native checkpoint order."""
        horizon, width = self.contract["action_horizon"], self.contract["action_dim"]
        parameters = {"mode": mode}
        if mode == "rtc":
            condition = np.asarray(sampling["action_condition"], dtype=np.float32)
            weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
            beta = float(sampling["beta"])
            if condition.shape != (horizon, width) or not np.isfinite(condition).all():
                raise ValueError(f"RTC condition must be finite {(horizon, width)}")
            if weights.shape != (horizon,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
                raise ValueError("RTC weights must be finite horizon values in [0, 1]")
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError("RTC beta must be finite and positive")
            parameters.update(
                action_condition=condition,
                condition_weights=weights,
                beta=beta,
            )
        elif mode == "paint":
            delay = sampling["delay_steps"]
            prefix = np.asarray(sampling["action_prefix"], dtype=np.float32)
            if type(delay) is not int or not 0 < delay < horizon:
                raise ValueError("PAINT requires 0 < delay_steps < horizon")
            if prefix.shape != (delay, width) or not np.isfinite(prefix).all():
                raise ValueError("PAINT prefix must be finite with the declared delay and action width")
            condition = np.zeros((horizon, width), dtype=np.float32)
            condition[:delay] = prefix
            parameters.update(action_condition=condition, delay_steps=delay)
        elif mode == "aac":
            samples = sampling["num_samples"]
            if type(samples) is not int or samples <= 1:
                raise ValueError("AAC num_samples must be an integer greater than one")
            parameters["num_samples"] = samples
        elif mode == "autohorizon":
            if set(sampling) - {"mode"}:
                raise ValueError("AutoHorizon does not accept sampling overrides")
        elif mode == "dvac":
            allowed = {
                "mode",
                "alpha",
                "tail_steps",
                "rolling_window_size",
                "min_execution_steps",
                "max_execution_steps",
            }
            if set(sampling) - allowed:
                raise ValueError("Unknown DVAC fields")
            alpha = float(sampling.get("alpha", 2.0))
            minimum = sampling.get("min_execution_steps", 1)
            maximum = sampling.get("max_execution_steps", horizon)
            window = sampling.get("rolling_window_size", 5)
            tail = sampling.get("tail_steps", 5)
            steps = self.contract["num_inference_timesteps"]
            if any(type(value) is not int for value in (minimum, maximum, window, tail)):
                raise ValueError("DVAC step/window settings must be integers")
            if (
                not math.isfinite(alpha)
                or alpha < 0
                or not 1 <= minimum <= maximum <= horizon
                or window < 1
                or not 1 < tail <= steps
            ):
                raise ValueError("Invalid DVAC bounds, alpha, window or denoising tail length")
            if self.window_size not in (None, window):
                raise ValueError("DVAC rolling_window_size cannot change before reset")
            parameters["tail_steps"] = tail

        return parameters

    def _calibrate_dvac(self, metadata, sampling):
        """Select an execution prefix from episode-local denoising variance."""
        horizon = self.contract["action_horizon"]
        alpha = float(sampling.get("alpha", 2.0))
        minimum = sampling.get("min_execution_steps", 1)
        maximum = sampling.get("max_execution_steps", horizon)
        window = sampling.get("rolling_window_size", 5)
        variance = np.asarray(metadata["variance"], dtype=np.float64)
        if variance.shape != (horizon,) or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("Invalid denoising variance")
        cold = not self.history
        calibration = variance if cold else np.concatenate(tuple(self.history))
        mean, std = float(calibration.mean()), float(calibration.std())
        threshold = mean + alpha * std
        indices = np.flatnonzero(variance > threshold)
        crossing = int(indices[0]) if len(indices) else None
        execution = maximum if crossing is None else min(maximum, max(minimum, crossing))
        self.window_size = window
        self.history.append(variance.copy())
        while len(self.history) > window:
            self.history.popleft()
        metadata.update(
            execution_steps=execution,
            first_threshold_crossing=crossing,
            threshold=threshold,
            rolling_mean=mean,
            rolling_std=std,
            alpha=alpha,
            min_execution_steps=minimum,
            max_execution_steps=maximum,
            rolling_window_size=window,
            rolling_states=len(self.history),
            cold_start=cold,
            cold_start_policy="current_variance_bootstrap",
            method="denoising_variance_adaptive_chunking",
            source="arxiv:2606.03847v1",
        )
        return metadata
