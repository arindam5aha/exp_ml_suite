# Code Samples: Data Utilities, Neural Network Blocks, and MOT Interface

This repository contains reusable Python components extracted from a larger machine-learning and laboratory-control workflow:

- Data handling helpers for PyTorch pipelines.
- Neural network classes (MLP and Gaussian MLP).
- A Qt-threaded interface that connects a RunBuilder/LabVIEW-style control stack with ML logic over TCP.

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [What Each Module Does](#what-each-module-does)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Module Reference](#module-reference)
7. [Data Formats](#data-formats)
8. [Known Constraints and Platform Notes](#known-constraints-and-platform-notes)
9. [Development Notes](#development-notes)

## Repository Layout

```text
.
|- data_handling.py      # Data wrappers, transforms, torch/NumPy conversion, HDF5 save/load
|- nn_classes.py       # MLP and GaussianMLP model definitions + initialization utilities
|- mot_interface.py    # QThread interface between GUI/run control and ML process
|- remote_interface.py # Length-prefixed TCP messaging layer for distributed communication
`- README.md
```

## What Each Module Does

### `data_handling.py`

Provides:

- `data_transformer`: feature scaling/normalization wrapper around scikit-learn transformers.
- `data_wrapper`: tuple-based in-memory dataset container with optional preprocessing and buffer support.
- `torch_it`, `torch_it_`, `numfy`: conversion helpers between Python/NumPy and PyTorch.

Typical use case: manage small-to-medium experiment rollouts or batches without introducing a full dataset class.

### `nn_classes.py`

Provides:

- `MLP`: configurable feed-forward network with per-layer activation selection and optional dropout.
- `GaussianMLP`: latent distribution head (`mean`, `std`, `sample`) with reparameterized sampling.
- Utilities for activation lookup, layer initialization, and distribution construction.

Typical use case: control, RL, latent-variable models, or compact supervised models.

### `mot_interface.py`

Provides:

- `RBMLInterface(QThread)`: event-driven communication bridge between machine-learning code and an external run-control stack.
- TCP-based flag protocol (`<sys_info>`, `<exec>`, `<acq>`, `<scan_img>`, etc.).
- Runtime helpers for run editing, image acquisition, lock checks, and compressed run-data persistence.

Typical use case: online optimization/control loop where experimental runs are configured by ML outputs and observations/rewards are returned to the model.

### `remote_interface.py`

Provides:

- `Receiver`: manages one or more TCP peer connections with framed-message protocol.
- `PeerConnection`: represents a single framed-message TCP connection (8-byte length prefix + UTF-8 payload).
- `SocketWrapper`: thin wrapper around Python's socket module with reusable defaults.
- `DaemonThread`: cooperative thread termination via event-based halt signaling.
- Symmetric `<CC>` handshake protocol for peer synchronization.

Typical use case: distributed communication between external control/monitoring processes and Python ML workflows.

## Requirements

### Core Python Dependencies

- Python 3.10+
- `torch`
- `numpy`
- `h5py`
- `scikit-learn`
- `PySide2`

### Additional Runtime Dependencies (for MOT interface)

`mot_interface.py` imports external modules that are not included in this repository:

- **`pueye_cam`** ([GitHub repository](https://github.com/arindam5aha/pueye_cam)): Python interface for IDS uEye industrial cameras. Provides camera initialization, configuration, and image capture using the `pyueye` library. Supports external trigger modes and image visualization with OpenCV and Matplotlib. Key features include exposure control, pixel clock configuration, frame rate management, and NumPy array image retrieval. Install with: `pip install pyueye numpy opencv-python matplotlib`

The `tcp_server` module is now included in this repository and no longer an external dependency.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell

pip install torch numpy h5py scikit-learn PySide2
```

If you are only using `data_handling.py` and `nn_classes.py`, `PySide2` and hardware/network dependencies are optional.

## Quick Start

### 1) Build and run an MLP

```python
import torch
from nn_classes import MLP

model = MLP(
		input_size=12,
		output_sizes=3,
		hidden_sizes=[128, 128],
		hidden_activation='relu',
		output_activation='identity'
)

x = torch.randn(32, 12)
y = model(x)
print(y.shape)  # torch.Size([32, 3])
```

### 2) Use `data_wrapper` for mini in-memory datasets

```python
import numpy as np
from data_handling import data_wrapper

dw = data_wrapper(
		name='Transition',
		data_keys=['obs', 'act', 'rew'],
		preprocess=False,
		buffer_size=2000
)

obs = np.random.randn(100, 8)
act = np.random.randn(100, 2)
rew = np.random.randn(100, 1)

dw.wrap((obs, act, rew))
batch = dw.sample(16)
all_data = dw.unwrap()  # namedtuple of stacked torch tensors
```

### 3) Save and reload wrapped data via HDF5

```python
dw.save('run_data.h5')
restored = dw.read('run_data.h5', store=False)
```

## Module Reference

### `data_handling.py`

### `class data_transformer(method='standard', bounds=None)`

Wrapper around scikit-learn preprocessing objects.

- Supported `method`: `standard`, `minmax`, `normalize`
- If `bounds` is provided, transformer is pre-fit using lower/upper bounds.
- Call instance directly to transform input.
- Use `.inv(...)` for inverse transform (not available for `normalize`).

### `class data_wrapper(...)`

Important constructor arguments:

- `name`: namedtuple type name for stored samples.
- `data_keys`: fields for each sample tuple.
- `preprocess`: whether to apply per-field transform during `wrap`.
- `buffer_size`: cap for rolling buffer used by `append`.

Common methods:

- `wrap((arr1, arr2, ...))`: ingest tuple of aligned arrays/lists.
- `append(...)`: append with buffer-size control.
- `sample(size)`: random samples from all stored data.
- `unwrap(buffer=False)`: stacked namedtuple of tensors.
- `save(file_name)`: write current full dataset to HDF5.
- `read(file_name, store=False)`: read HDF5 datasets by key.
- `flush(all=False)`: clear buffer only (default) or all data.

Utility functions:

- `torch_it(data, device=cpu)`
- `torch_it_(input)` (auto-selects CUDA if available)
- `numfy(data)`

### `nn_classes.py`

### `class MLP(nn.Module)`

Highlights:

- Flexible output definition via `int`, `list`, or `dict`.
- Per-hidden/per-output activations via string or list.
- Optional dropout.
- Optional return of final hidden state (`return_last_hidden=True`).
- Optional tuple input concatenation (`concat_inputs=True`).

Output behavior:

- Single output head: tensor.
- Multiple heads with list sizes: tuple of tensors.
- Multiple heads with dict sizes: dict keyed by provided output names.

### `class GaussianMLP(MLP)`

Outputs latent dictionary:

- `sample`: reparameterized sample.
- `mean`: latent mean.
- `std`: positive latent std.

Useful flags:

- `clamp_latent`, `mu_range`, `std_range`
- `batch_norm_mean`
- `independent_normal`

Helper functions:

- `activation_from_string(...)`
- `layer_initialiser(...)`
- `normal_dist(...)`
- `get_dist(state)`
- `reparameterize(mean, std, logvar=False)`

### `mot_interface.py`

### `class RBMLInterface(QThread)`

Primary responsibility:

- Coordinate run-generation, execution signaling, trace/image capture, and network communication between GUI/control and ML side.

Selected Qt signals:

- `load_sess(dict)`
- `plot_ch(list, bool)`
- `push_run(bool)`
- `exit()`
- `reset(bool, bool)`

Selected methods:

- Lifecycle: `start_server`, `stop_server`, `start_ml`, `stop_ml`, `quit`
- Run control: `open_file`, `run_params`, `make_run`, `drop`
- Acquisition: `acq_trace`, `scan_capture_imaging`, `get_img`, `get_obs`
- Health/safety: `check_lock`, `clear_lv_buffer`
- Utilities: `save_bz2`, `load_bz2`, `topk`, `get_datetime`

### Flag Protocol (ML -> Interface)

The communication thread dispatches actions based on string flags such as:

- `<sys_info>`
- `<clr>`
- `<drop>`
- `<acq>`
- `<acq_ref>`
- `<exec>`
- `<reset>`
- `<check>`
- `<probe>`
- `<imaging>`
- `<img_ref>`
- `<scan_img>`
- `<jeitgem>`
- `<stop>`

### `remote_interface.py`

### `class Receiver(host, port)`

Manages one or more TCP peer connections using frame-based message framing.

Key methods:

- `start_server()`: begin accepting connections in a background thread.
- `initiate_connection(host, port, timeout)`: create an outbound connection and perform handshake.
- `conn_send(msg)`: broadcast a UTF-8 message to all active connections (returns bytes sent).
- `conn_read()`: read and concatenate one message from each active connection.

Attributes:

- `listening_thread`: internal `DaemonThread` managing the accept loop. Access `request_stop()` to gracefully stop.
- `peers`: list of active `PeerConnection` instances.

### `class PeerConnection`

Represents a single framed-message TCP connection.

Methods:

- `send(msg)`: send UTF-8 message with 8-byte ASCII length prefix (returns payload bytes sent, or -1 on error).
- `read()`: read one framed UTF-8 message (blocking).

### Message Frame Format

Messages follow a simple framing protocol:

- **8-byte ASCII header**: zero-padded decimal length of the payload.
- **Payload**: UTF-8 encoded message string (0+ bytes).

Example:

```
Frame for message "hello" (5 bytes):
Header:  "00000005"
Payload: "hello"
```

### Thread Lifecycle

The `DaemonThread` class provides cooperative termination:

- `request_stop()`: signal the thread to cleanly exit.
- `should_halt()`: check termination flag.

## Data Formats

### HDF5 via `data_wrapper.save(...)`

Each `data_key` is written as a dataset in the output HDF5 file.

Example dataset keys:

- `obs`
- `act`
- `rew`

### Compressed run data via `save_bz2(...)`

`mot_interface.py` stores runtime logs/data in bz2-compressed pickle format.

## Known Constraints and Platform Notes

- `start_ml` currently launches with `gnome-terminal`, which is Linux-specific.
- The interface assumes external run files and channel schemas compatible with the surrounding RunBuilder/LabVIEW ecosystem.
- Camera-dependent methods require compatible hardware and a working `pueye_cam` integration.
- Lock-check workflow assumes a second TCP listener/service is available.

## Development Notes

- The repository is organized as code samples and building blocks rather than a packaged library.
- For production use, consider adding:
	- Unit tests for `data_wrapper`, transform inversion, and `nn_classes` output contracts.
	- Type hints on public methods in `mot_interface.py`.
	- Platform-agnostic process launching for ML subprocess startup.
	- A `requirements.txt` or `pyproject.toml` for reproducible environments.

---

Author metadata embedded in source files:

- Arindam Saha, ANU (2025)
- GitHub: `arindam5aha`

