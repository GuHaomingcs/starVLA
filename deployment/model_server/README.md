# Policy server

Choose the entry point that matches your client:

| Entry point | Client contract |
| --- | --- |
| `serve.py --config ...` | Explicit deployment recipe, checked metadata, per-connection sessions and optional flow sampling |
| `server_policy.py --ckpt_path ...` | Existing evaluation WebSocket clients |
| `server_policy_gr00t_zmq.py --ckpt_path ...` | Isaac-GR00T N1.6 ZMQ clients |

All paths reuse the checkpoint's native model and training transforms. The configured
WebSocket guide comes first; existing evaluation and ZMQ instructions follow below.

## Configured native policy serving

`serve.py` exposes a checkpoint through StarVLA's existing WebSocket/MessagePack
transport. A deployment recipe makes observation order, action meaning and supported
sampling modes explicit. It requires neither a robot SDK nor a controller connection.

From the StarVLA source root, in its model environment:

```bash
python -m deployment.model_server.serve --config /path/to/deployment.yaml
```

Install the [model dependencies](../../docs/starVLA_guideline.md#0-installation),
plus `websockets>=14`, `msgpack`, `PyYAML` and `opencv-python`. Preserve the weight
file's sibling run metadata (`config.yaml` and `dataset_statistics.json`). Checkpoint
loading and normalization continue to use the native framework and registered
training transforms.

### Recipe

Example for a RoboDojo PI-v3 joint checkpoint:

```yaml
host: 127.0.0.1
port: 8500
checkpoint_path: /path/to/run/checkpoints/steps_100000_pytorch_model.pt
device: cuda
use_bf16: true
framework: QwenPI_v3
pi_v3_forward: canonical
unnorm_key: arx_x5
action_horizon: 50
action_type: joint
action_semantics: absolute_joint_position
include_state: true
state_type: joint
camera_names: [cam_head, cam_left_wrist, cam_right_wrist]
image_size: [224, 224]
enabled_sampling_modes: [rtc, paint, aac, autohorizon, dvac]
config_overrides:
  - framework.qwenvl.base_vlm=/path/to/Qwen3-VL-4B-Instruct
```

`action_indices` describes canonical action order as indices into native output.
`state_indices` describes native state order as indices into canonical input.
Each is a full permutation, identity when omitted. They are metadata for the client;
the server accepts and returns arrays in native checkpoint order. EEF recipes also
supply an `eef` mapping describing rotation encoding, scaling and semantics.
The server checks checkpoint identity, horizon, state use and sampling capabilities
at startup. Client geometry and robot grouping are not model-side responsibilities.

`config_overrides` uses the native OmegaConf dotlist loader. Supply asset paths or
an explicitly selected supported attention backend. No attention fallback is selected
silently. FAST needs `framework.action_model.fast_tokenizer_path` (or the explicit
`STARVLA_FAST_TOKENIZER` environment binding) and its matching action-token VLM.

### Protocol and lifecycle

The initial binary frame contains native metadata plus `protocol_version: 1`,
`supports_reset: true` and `serving_contract`. Arrays use the native
`tools/msgpack_numpy.py` encoding. This is not the third-party `msgpack_numpy` format.

Before inference, send:

```python
{"type": "reset", "request_id": "r0", "session_id": "episode-1"}
```

The response echoes both IDs. Each connection owns its session and DVAC history;
reset clears that history. It does not reload weights or reseed flow noise. The
included frameworks do not cache action chunks or model episode state. Add explicit
reset support before integrating a future recurrent framework.

Inference messages contain:

```python
{
    "type": "infer", "request_id": "r1", "session_id": "episode-1",
    "payload": {
        "examples": [{"image": [head_rgb, left_rgb, right_rgb],
                      "lang": "Pick up the block.", "state": raw_state}],
        "unnorm_key": "arx_x5",
        "sampling": {"mode": "default"},
    },
}
```

Configured serving accepts one observation per request. Images are HWC `uint8` RGB
in recipe order, resized with `INTER_AREA`. State is `(1, state_dim)` in raw units;
omit it for a stateless recipe. Training transforms normalize state and action
conditions and unnormalize output. A successful response has `type: inference_result`
and `data.actions` shaped `(B, horizon, action_dim)`, where `B` is one except for AAC.
Sampler metadata travels in `data.paint`, `data.autohorizon` or `data.dvac`.
Errors return `ok: false` with an error message; clients must not execute a failed
response. A session mismatch or an unsupported mode is an error.

### Source layout

| File | Responsibility |
| --- | --- |
| `serve.py` | Recipe loading and service startup |
| `serving_contract.py` | Checkpoint/recipe validation and observation contract |
| `policy_wrapper.py` | Native model loading, training transforms and prediction |
| `policy_norm_processor.py` | Apply/unapply registered training normalization |
| `sampling_session.py` | Per-connection reset, sampler request validation and DVAC history |
| `tools/websocket_policy_server.py` | Transport and structured RPC responses |
| `../../starVLA/model/modules/action_model/runtime_sampling.py` | Flow-head sampling hooks |

Transport errors and model failures remain errors. There is no automatic attention
backend replacement or zero-action substitution. AAC requests must specify their
candidate count. DVAC retains documented defaults: `alpha=2`, `tail_steps=5`,
`rolling_window_size=5`, and execution bounds `[1, action_horizon]`; validation
still requires the tail to fit the loaded head's denoising steps.

Legacy clients remain compatible with the existing unconfigured wrapper/server.
They do not acquire configured reset or specialized sampling implicitly.

### Sampling

| Mode | Request fields | Model work |
| --- | --- | --- |
| `default` | None | Original checkpoint inference |
| `rtc` | `action_condition: (H,D)`, `condition_weights: (H,)`, positive `beta` | Guided flow integration |
| `paint` | `action_prefix: (delay,D)`, `delay_steps` | Prefix inversion and flow continuation |
| `aac` | `num_samples > 1` (required) | Independent candidate generation; client selects using its geometry |
| `autohorizon` | None | Attention-based horizon estimates |
| `dvac` | `tail_steps`, `alpha`, `rolling_window_size`, execution bounds | Denoising variance and per-session prefix calibration |

Specialized modes are opt-in for QwenGR00T/QwenPI_v3 joint deployments. OFT and
FAST support default inference. EEF condition conversion is not implemented.
AutoHorizon requires self-attention and at least three denoising steps; PI-v3 must
use canonical forwarding, since legacy forwarding uses cross-attention. DVAC requires
at least two and rejects a tail longer than the available steps. Enabling a mode
never substitutes a different sampler when its requirements are absent.

The specialized hooks preserve original default sampling. FAST validates generated
action tokens before decoding and raises on invalid output instead of returning
zero actions. AutoHorizon attribution and its license accompany the implementation
under `starVLA/model/modules/action_model/`.

### Checking a configured deployment

Use a client that implements the lifecycle above: read the handshake, verify the
expected contract, reset the session, then send one observation. Verify finite
`(B, horizon, action_dim)` output in environment units, reset behavior and each
explicitly enabled sampler before connecting a controller. OFT and deterministic
FAST can be checked with repeated inputs; flow models sample fresh noise.

The contribution was also checked with small real action heads and training
normalization transforms, without pretrained weights. Those external regression
fixtures cover Euler equivalence, guided modes, inference-mode hooks, invalid FAST
tokens, explicit attention selection and normalization round trips. They are kept
outside this model source checkout. These checks do not establish manipulation task
success or support for every checkpoint.

## Existing evaluation services

### Start the server

```bash
your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```

### GR00T-protocol server (ZMQ) — drop-in for Isaac-GR00T clients

For robot stacks already speaking the Isaac-GR00T N1.6 `PolicyServer` ZMQ
protocol (e.g. the GR00T-WBC-Bridge deploy script for the Unitree G1 whole-body
stack), serve the same checkpoint over that protocol instead of websockets:

```bash
python deployment/model_server/server_policy_gr00t_zmq.py \
    --ckpt_path ${your_ckpt} \
    --port 5555 \
    --use_bf16
```

The existing GR00T client keeps working unchanged (same msgpack
`__ndarray_class__` codec, same `ping` / `reset` / `get_action` endpoints).
The server:

- flattens the client's named state dict (`{"left_arm": (1, n, 7), ...}`)
  into the flat state vector, **in the checkpoint DataConfig's `state_keys`
  order**, taking the latest history frame per group;
- splits the unnormalized flat action chunk `(1, T, D)` back into named groups
  (`{"left_arm": (1, T, 7), ...}`) using the DataConfig's `action_keys` /
  `action_key_dims`.

So bridging a new embodiment needs no protocol code — only a registered
DataConfig whose state/action key names match what the client sends/consumes.
`get_modality_config` returns the full key/dim contract for handshake checks.
Components: `gr00t_obs_adapter.py` (conversion) +
`tools/zmq_policy_server.py` (transport). Requires `pyzmq`.

Caveats:

- Only the N1.6 "custom" codec is implemented (GR00T-WBC-Bridge default
  `--server_codec custom`); the N1.7 msgpack-numpy codec is not.
- State/video history stacking beyond "latest frame" is not supported —
  matches the single-frame `examples` schema used across starVLA eval clients.

### Connect to server for debug

```bash
python deployment/model_server/debug_server_policy.py

# use server_policy.py in your eval client by referencing debug_server_policy.py
```

---

## Normalization and checkpoint metadata

The server returns `data.actions` in environment units. Clients must not unnormalize
these values again. `PolicyNormProcessor` uses the checkpoint's registered training
transforms; `PolicyServerWrapper` loads the model and returns the converted chunk.

For existing evaluation clients:

1. Read `action_chunk_size` from the server metadata.
2. Pass the matching `unnorm_key` with each prediction request.
3. Consume `response["data"]["actions"]`.

An unconfigured wrapper can serve a multi-dataset checkpoint by selecting a processor
lazily from each request's `unnorm_key`; metadata lists `available_unnorm_keys`.
Configured serving binds one key in its deployment recipe and rejects another key.

```python
metadata = client.get_server_metadata()
response = client.predict_action({"examples": examples, "unnorm_key": "oxe_bridge"})
actions = np.asarray(response["data"]["actions"][0])  # (horizon, action_dim)
```

The wrapper reads the native horizon from either `action_model.action_horizon` or
`action_model.future_action_window_size + 1`, according to the checkpoint format.
Clients should use the advertised horizon instead of reconstructing it themselves.
