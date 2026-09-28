"""Explicit deployment identity and checkpoint observation contract."""

import cv2
import numpy as np


def _sampling_capabilities(policy, framework):
    """Inspect the loaded action head without enabling optional samplers."""
    modes = ["default"]
    steps = None
    forward = None
    if framework in {"QwenGR00T", "QwenPI_v3"}:
        head = policy._framework.action_model
        steps = int(head.num_inference_timesteps)
        if framework == "QwenPI_v3":
            forward = "canonical" if head.model.config.use_canonical_forward else "legacy"
        modes += ["rtc", "paint", "aac"]
        if steps >= 2:
            modes.append("dvac")
        # Legacy PI-v3 sends VLM context through attn1 despite its module flag.
        # AutoHorizon needs action self-attention actually executed by forward().
        if (
            steps >= 3
            and forward != "legacy"
            and any(n.endswith("attn1") and not m.is_cross_attention for n, m in head.model.named_modules())
        ):
            modes.append("autohorizon")
    return modes, steps, forward


def configure_contract(policy, config):
    """Validate a recipe against loaded weights and build handshake metadata."""
    model = policy._model_cfg
    framework = str(model["framework"]["name"])
    horizon = policy._action_chunk_size
    if config.get("framework", framework) != framework:
        raise ValueError("Configured framework differs from checkpoint")
    if config["action_horizon"] != horizon:
        raise ValueError("Configured action_horizon differs from checkpoint")
    include_state = config["include_state"]
    if type(include_state) is not bool:
        raise ValueError("include_state must be an explicit boolean")
    trained_state = model["datasets"]["vla_data"].get("include_state")
    if trained_state is not None and bool(trained_state) != include_state:
        raise ValueError("include_state differs from checkpoint training configuration")
    proc = policy.get_norm_processor(config["unnorm_key"])
    modes, steps, forward = _sampling_capabilities(policy, framework)
    if forward is not None and config.get("pi_v3_forward", forward) != forward:
        raise ValueError("PI-v3 forward mode differs from the deployment recipe")
    requested = config.get("enabled_sampling_modes", [])
    if not isinstance(requested, list) or set(requested) - set(modes):
        raise ValueError("Requested sampling modes are unsupported by the checkpoint")
    if requested and config["action_type"] != "joint":
        raise ValueError("Conditioned EEF sampling is not implemented")
    if config["action_type"] not in {"joint", "ee"} or config["state_type"] not in {"joint", "ee"}:
        raise ValueError("action_type/state_type must be joint or ee")
    if config["action_type"] == "joint" and config["action_semantics"] != "absolute_joint_position":
        raise ValueError("Joint actions require absolute_joint_position semantics")
    if config["action_type"] == "ee" and config["eef"]["semantics"] != config["action_semantics"]:
        raise ValueError("EEF conversion and action semantics must match")
    cameras = config["camera_names"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(c, str) or not c for c in cameras)
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError("camera_names must be unique ordered names")
    size = config["image_size"]
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("image_size must be positive [width, height]")
    result = {
        k: config[k]
        for k in ("checkpoint_path", "action_type", "include_state", "state_type", "camera_names", "image_size")
    }
    for key, width in [("action_indices", proc.action_dim), ("state_indices", proc.state_dim)]:
        order = config.get(key, list(range(width)))
        if sorted(order) != list(range(width)) or any(type(i) is not int for i in order):
            raise ValueError(f"{key} must be a full dimension permutation")
        result[key] = order
    result.update(
        framework=framework,
        action_horizon=horizon,
        action_dim=proc.action_dim,
        state_dim=proc.state_dim,
        unnorm_key=proc.unnorm_key,
        sampling_modes=["default", *sorted(set(requested) - {"default"})],
        num_inference_timesteps=steps,
        action_semantics=config["action_semantics"],
        eef=config.get("eef", {}),
        image_color_order="rgb",
        state_normalization="training_transform",
        action_output="unnormalized_env",
    )
    if forward is not None:
        result["pi_v3_forward"] = forward
    return result


def validate_examples(examples, contract):
    if not isinstance(examples, list) or len(examples) != 1:
        raise ValueError("Configured real-time serving requires one observation")
    example = examples[0]
    if not isinstance(example["lang"], str):
        raise ValueError("lang must be a string")
    images = example["image"]
    if len(images) != len(contract["camera_names"]):
        raise ValueError("Image count differs from deployment camera order")
    width, height = contract["image_size"]
    for image in images:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Images must be HWC uint8 RGB arrays")
    if contract["include_state"]:
        state = np.asarray(example["state"])
        if state.shape != (1, contract["state_dim"]) or not np.isfinite(state).all():
            raise ValueError("Raw state must match checkpoint state dimensions")
    elif "state" in example:
        raise ValueError("This deployment does not accept state input")

    return [{**example, "image": [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]}]
