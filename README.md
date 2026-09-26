# GoGAC: Group-Aware Communication for GoMARL

PyTorch implementation of **GoGAC**, a Group-Aware Communication module integrated
into GoMARL for cooperative multi-agent target coverage. The project extends the
automatic grouping mechanism of GoMARL with communication restricted to agents in
the same learned group.

The main experiment uses the MATE distributed target-coverage environment:

- fixed cameras control viewing direction and field of view;
- mobile targets move through a workspace containing optional obstacles;
- each camera receives a local, partially observable observation;
- the team is trained with centralized training and decentralized execution (CTDE).

The primary metric is the **Mean Coverage Rate (MCR)**:

\[
\mathrm{MCR}=\frac{1}{T}\sum_{t=1}^{T}\frac{1}{m}\sum_{j=1}^{m}C_{j,t},
\]

where $C_{j,t}=1$ when target $j$ is covered by at least one camera at timestep
$t$.

## Repository layout

```text
GoMARL-MATE-GAC/
├── GoMARL/                 # GoMARL training framework and GoGAC integration
│   ├── src/controllers/    # Basic, group, and GAC controllers
│   ├── src/modules/agents/ # Recurrent agents and GAC communication agent
│   ├── src/learners/       # Group learner and TD(lambda) training
│   ├── src/envs/           # MATE, SMAC, and GFootball adapters
│   └── src/config/         # Algorithm and environment configurations
└── MATE-main/              # MATE environment package
```

## Method overview

At timestep $t$, each camera encodes its observation history with a GRU:

\[
h_i^t=\mathrm{GRU}(o_i^t,h_i^{t-1}).
\]

GoMARL dynamically partitions the agents into groups. GoGAC uses that grouping to
build a masked multi-head attention layer. Camera $i$ can attend to camera $j$
only when both cameras belong to the same group. The resulting group message is
combined with the local representation using a learned gate:

\[
\beta_i=\sigma(w_g^\top[h_i^t\Vert m_i^t]+b_g),\qquad
h_{i,\mathrm{fused}}^t=(1-\beta_i)h_i^t+\beta_i m_i^t.
\]

The fused representation is used by the individual Q-network and the monotonic
group mixer. Training combines TD learning with GoMARL's grouping regularizers:

\[
\mathcal{L}=\mathcal{L}_{TD}+\lambda_{lasso}\mathcal{L}_{lasso}
								 +\lambda_{SD}\mathcal{L}_{SD}.
\]

The implementation also logs the average communication gate (`comm_beta_mean`)
and the ratio of isolated agents (`isolated_agent_ratio`).

## Installation

The repository contains two packages with different dependency generations. Use
the Python version required by the MATE package for the MATE experiments, and
install MATE as an editable local package.

From the repository root:

```bash
cd MATE-main
python -m pip install -e .
cd ../GoMARL
python -m pip install -r requirements.txt
```

On Linux, the legacy SMAC and GFootball setup scripts can be run when those
environments are needed:

```bash
bash install_sc2.sh
bash install_gfootball.sh
```

The MATE configuration used by GoMARL is
`src/config/envs/mate.yaml`. It currently points to
`../MATE-main/mate/assets/MATE-4v6-0.yaml`, relative to the `GoMARL` directory.
Run commands from `GoMARL` so this relative path resolves correctly.

> **Dependency note:** `requirements.txt` contains the original GoMARL/PyMARL
> stack, while `MATE-main/pyproject.toml` contains the newer MATE stack. If pip
> reports incompatible pinned packages, use separate environments for the
> original SMAC/GFootball experiments and the MATE experiments, or install the
> MATE dependencies first and resolve any project-specific version conflict.

## Training GoGAC on MATE

The default command reproduces the GoGAC configuration on MATE(4v6-0):

```bash
cd GoMARL
python src/main.py --config=gac_group --env-config=mate
```

The default GoGAC run uses 4 parallel environments, a 4,000,000-step budget,
dynamic grouping, group-restricted communication, adaptive gating, and the
`group` value mixer. Main settings are in:

- `src/config/algs/gac_group.yaml` for GoGAC, grouping, optimizer, and training;
- `src/config/envs/mate.yaml` for the MATE environment;
- `src/config/default.yaml` for logging, checkpoint, and evaluation options.

### MATE(4v8-9)

To run the obstacle setting from the paper, edit `src/config/envs/mate.yaml`:

```yaml
env: mate
env_args:
	config: "../MATE-main/mate/assets/MATE-4v8-9.yaml"
```

Then run the same command. The MATE asset controls the number of cameras,
targets, and obstacles.

### Multiple seeds and GPUs

`run.sh` launches repeated runs in parallel on Linux/macOS. For example:

```bash
bash run.sh gac_group mate 4v6-0 "seed=1" 2 0,1 5
```

The script's third argument is passed as `env_args.map_name`, so for custom MATE
assets prefer the direct Python command above or update the script accordingly.

## Baselines and ablations

GoMARL without GAC can be run with the original grouping configuration:

```bash
python src/main.py --config=group --env-config=mate
```

The GAC-specific switches are defined in `src/config/algs/gac_group.yaml`:

| Setting | Purpose |
| --- | --- |
| `gac_use_gate: True` | Enables adaptive fusion of local and group representations |
| `gac_carry_fused_hidden: False` | Keeps the GRU state dedicated to observation history |
| `gac_comm_dim: 32` | Communication/message dimension |
| `gac_explicit_comm: False` | Selects the default implicit communication path |
| `lasso_alpha_start` | Strength of communication/group sparsity regularization |
| `sd_alpha` | Shape-diversity regularization weight |

For the ablations described in the paper, change one switch at a time:

- **Without adaptive gate:** set `gac_use_gate: False`.
- **Fused hidden variant:** set `gac_carry_fused_hidden: True`.

Keep the environment, seed, training budget, and evaluation settings identical
when comparing curves.

## Evaluation, logging, and checkpoints

Enable TensorBoard in `src/config/default.yaml`:

```yaml
use_tensorboard: True
```

Logs are written under `results/tb_logs`. To evaluate a saved checkpoint without
training, set:

```yaml
checkpoint_path: "results/models/<checkpoint>"
evaluate: True
```

Models and optimizer state are saved under `results/models/` when
`save_model: True`. The group structure is stored as `group.npy`, while mixer
weights are stored as `mixer.th`.

## Troubleshooting

- **`ModuleNotFoundError: mate`:** install `MATE-main` with `python -m pip install -e .`
	and run from the `GoMARL` directory.
- **MATE config not found:** check that `src/config/envs/mate.yaml` points to a
	valid asset under `MATE-main/mate/assets/`.
- **CUDA unavailable:** set `use_cuda: False` in `src/config/default.yaml` or
	provide a CUDA-enabled PyTorch installation.
- **Hard-coded local path:** `src/main.py` contains a developer-specific MATE
	path. Editable installation normally makes it unnecessary; if imports still
	fail, replace that path with the absolute path to this repository's
	`MATE-main` directory.

## Related work

- [GoMARL: Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2310.17679)
- [MATE: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control](https://github.com/XuehaiPan/MATE)
- [PyMARL](https://github.com/oxwhirl/pymarl)
- [PyMARL2](https://github.com/hijkzzz/pymarl2)
- [SMAC](https://github.com/oxwhirl/smac)
- [GFootball](https://github.com/google-research/football)
