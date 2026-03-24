# 🪰 Chat with a Fruit Fly Brain

> Talk to 138,000 neurons. Ask a fly to walk, taste sugar, or run for its life.

A whole-brain computational model of the adult *Drosophila melanogaster* (fruit fly), built on the [FlyWire](https://flywire.ai/) connectome (~138k neurons, ~5M synapses). Translates natural language into neural stimulations, runs a leaky integrate-and-fire (LIF) simulation across the entire brain, and describes what the fly would *do*.

Based on [Shiu et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1) — *"A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing"*.

## ✨ What Can You Do?

| Stimulus | What Happens |
|----------|-------------|
| 🍬 Sweet taste | Fly extends proboscis → feeding |
| 💀 Bitter taste | Fly retracts → rejection |
| 🦶 Walk forward | Coordinated leg movement |
| 🔙 Walk backward | Moonwalker retreat |
| 🚀 Escape | Giant fiber activation → emergency takeoff |
| 🧹 Groom | Antenna cleaning with front legs |
| 👁 Looming shadow | Visual threat → triggers escape |
| 👂 Sound | Auditory processing via Johnston's organ |
| 👃 Danger smell | Olfactory avoidance (geosmin detection) |

**Combine them:** Walk + looming shadow, sweet + bitter (watch the fly get confused)

**Learning:** The fly has a mushroom body with dopamine-modulated plasticity. Punish it (⚡ electric shock) or reward it (🍰) and it remembers!

## 🚀 Quick Start

### 1. Clone (includes connectome data via Git LFS)

```bash
git clone https://github.com/lixiang1076/fly-brain.git
cd fly-brain
```

> 💡 Large data files (~200 MB) are stored with [Git LFS](https://git-lfs.github.com/). Make sure you have `git-lfs` installed. If data files appear as small pointer files, run `git lfs pull`.

### 2. Install dependencies

```bash
conda env create -f environment.yml
conda activate brain-fly
```

### 3. Chat with the fly!

**Interactive mode (recommended):**
```bash
python fly_chat.py
```

Then type in Chinese or English:
```
🪰 > 给果蝇尝甜的
🔬 理解: 给果蝇尝甜的
   开始仿真 13.8万个神经元...

==================================================
🪰  果蝇大脑仿真完成！
    仿真时长: 0.1秒
    活跃神经元: 324/138639
    总脉冲数: 1571
==================================================

🧠 果蝇的反应:
    👅 果蝇伸出了口器（proboscis）！它在积极进食，看起来很享受。

📊 关键输出神经元活动:
    MN9_left              80.0 Hz  ████████████████  (口器运动 (进食))
    MN9_right             60.0 Hz  ████████████  (口器运动 (进食))
```

**Command-line mode (JSON output):**
```bash
# Single stimulus
python chat_with_fly.py --stim taste_sweet --duration 0.1 --pretty

# Combo: walking + visual threat
python chat_with_fly.py --stim walk_forward,vision_looming --freq 100,200 --duration 0.1 --pretty

# Silence a sense (blind the fly, then make it walk)
python chat_with_fly.py --stim walk_forward --silence vision_looming --duration 0.1 --pretty

# Stimulate by neuron ID directly
python chat_with_fly.py --neuron-ids 720575940622838154,720575940632499757 --freq 200 --duration 0.1 --pretty
```

## 🧠 Learning & Memory

The fly has a **mushroom body** — the insect equivalent of associative memory. It supports dopamine-modulated learning:

```
🪰 > 给果蝇尝甜的     ← present a stimulus
🪰 > 电击              ← punish! (PPL1 dopamine neurons fire)
⚡ 果蝇学会了：上次的刺激 = 危险！

🪰 > 给果蝇尝甜的     ← same stimulus again
🧠 果蝇记得这个气味是危险的！强烈回避反应  ← learned avoidance!

🪰 > 记忆              ← check what the fly has learned
🪰 > 失忆              ← wipe its memory (amnesia)
```

This mirrors real fly learning: **Kenyon Cells (KC)** encode stimulus identity, **Dopamine Neurons (DAN)** carry reward/punishment signals, and **KC→MBON** synaptic weights are modified accordingly.

- 4,133 Kenyon Cells (KCγ + KCαβ)
- 96 Mushroom Body Output Neurons (MBON)
- 307 PAM dopamine neurons (reward)
- 24 PPL1 dopamine neurons (punishment)

## 📁 Project Structure

```
fly-brain/
├── fly_chat.py              # 🎮 Interactive chat interface (start here!)
├── chat_with_fly.py         # 🔧 CLI simulation wrapper (JSON output)
├── dopamine_learning.py     # 🧠 Mushroom body learning (KC→MBON plasticity)
├── fast_learning.py         # ⚡ Fast post-hoc learning (no recompile needed)
├── neuron_atlas.json        # 🗺️ Stimulus→neuron mappings & output definitions
├── main.py                  # 📊 Multi-framework benchmark runner
├── environment.yml          # 📦 Conda environment
├── code/
│   ├── benchmark.py         # Benchmark orchestrator
│   ├── paper-phil-drosophila/
│   │   ├── model.py         # Core LIF model (Brian2)
│   │   ├── utils.py         # Analysis utilities
│   │   ├── example.ipynb    # Tutorial notebook
│   │   └── figures.ipynb    # Reproduce paper figures
│   ├── run_brian2_cuda.py   # Brian2/Brian2CUDA runner
│   ├── run_pytorch.py       # PyTorch runner
│   └── run_nestgpu.py       # NEST GPU runner
├── data/
│   ├── 2025_Completeness_783.csv      # Neuron list (FlyWire v783, LFS)
│   ├── 2025_Connectivity_783.parquet  # Synapse connectivity (LFS)
│   ├── mushroom_body_neurons.json     # MB neuron IDs
│   ├── flywire_annotations.tsv       # Neuron annotations (LFS)
│   └── archive/                       # Legacy v630 data (LFS)
└── scripts/
    ├── download_data.py     # Verify data files
    └── setup_WSL_CUDA.sh    # WSL2 + CUDA setup guide
```

## 📊 Benchmark: 4 Frameworks Compared

The same LIF model runs on four simulation backends:

| Framework | Backend | Notes |
|-----------|---------|-------|
| **Brian2** | C++ standalone (CPU) | Reference implementation |
| **Brian2CUDA** | CUDA (GPU) | GPU-accelerated Brian2 |
| **PyTorch** | CUDA (GPU) | Sparse tensor operations |
| **NEST GPU** | CUDA (custom kernel) | Requires separate build |

```bash
python main.py --pytorch --t_run 1 --n_run 1    # single backend
python main.py                                    # all backends, all durations
```

## 📦 Data

FlyWire connectome v783 (public release). Stored via **Git LFS**.

| File | Description | Size |
|------|-------------|------|
| `2025_Completeness_783.csv` | 138,639 neuron IDs + metadata | 3.3 MB |
| `2025_Connectivity_783.parquet` | ~5M synaptic connections + weights | 96 MB |
| `mushroom_body_neurons.json` | 4,133 KC + 96 MBON + 331 DAN neuron IDs | 120 KB |
| `flywire_annotations.tsv` | Neuron type annotations | 31 MB |
| `archive/` | Legacy v630 data (for paper figure reproduction) | 86 MB |

## ⚙️ Requirements

- **Minimum:** Python 3.10+, 8 GB RAM (Brian2 CPU backend)
- **Recommended:** NVIDIA GPU with CUDA 12.x (for PyTorch / Brian2CUDA backends)
- First simulation takes ~60–90s (Brian2 C++ compilation); subsequent runs reuse the compiled model

## 🔬 How It Works

1. **Input** — specify neurons to stimulate (natural language, stimulus keys, or FlyWire IDs)
2. **Mapping** — `neuron_atlas.json` resolves stimuli to specific FlyWire neuron IDs
3. **Simulation** — Brian2 runs a LIF simulation across all 138,639 neurons with 0.1ms timestep
4. **Analysis** — output neuron firing rates → behavioral predictions
5. **Learning** — mushroom body KC→MBON weights modified by dopamine signals

Neuron model parameters from [Kakaria & de Bivort (2017)](https://doi.org/10.3389/fnbeh.2017.00008):

| Parameter | Value |
|-----------|-------|
| Resting potential | −52 mV |
| Spike threshold | −45 mV |
| Membrane time constant | 20 ms |
| Synaptic time constant | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |

## 📜 Citation

If you use this code, please cite:

```bibtex
@article{shiu2023lif,
  title={A leaky integrate-and-fire computational model based on the connectome
         of the entire adult Drosophila brain reveals insights into sensorimotor processing},
  author={Shiu, Philip K and others},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.05.02.539144}
}
```

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

The FlyWire connectome data is subject to the [FlyWire Terms of Use](https://flywire.ai/).
