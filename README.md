# GNN Cost Model for TVM MetaSchedule

## Introduction

This repository contains the code for my final-year project at the University of Manchester (EEE, 2025-2026). The project investigates whether a graph neural network (GNN) can be used as a cost model for automatic tensor program tuning in TVM MetaSchedule, as an alternative to the default XGBoost-based approach.

The main idea is to represent scheduled TensorIR programs as graphs (with loops, computation blocks, and buffers as nodes) and train a GraphSAGE model to predict execution time. This predicted cost is then used to rank candidate schedules during the MetaSchedule search process.

## Files

- `tir_graph_builder_v2.py` — Converts a scheduled TensorIR program (IRModule) into a PyTorch Geometric graph. Extracts 16-dimensional node features and 32-dimensional graph-level features.
- `gnn_model_v2.py` — The GNN cost model (GraphSAGE architecture), training loop, ranking loss, Spearman evaluation, and data loading.

## Requirements

- Python 3.10+
- TVM 0.23.dev0 (built from source with CUDA support)
- PyTorch 2.5.1
- PyTorch Geometric
- NumPy

TVM needs to be compiled from source with the target GPU enabled. See https://tvm.apache.org/docs/install/from_source.html for instructions.

## How to Use

### Dataset

The training data is collected by running MetaSchedule with a dummy cost model, which measures scheduled TIR programs on the target GPU. Each measured sample is saved as a `.pt` file with the graph representation and log-runtime label.

The dataset directory should be organised by workload:
```
dataset_sched/
├── matmul_256/
│   ├── sample_0.pt
│   └── ...
├── conv3x3/
│   └── ...
```

### Training

```bash
python gnn_model_v2.py
```

This trains the model with default settings (lr=1e-3, 3 GraphSAGE layers, hidden dim 64). The best checkpoint is saved to `outputs/gnn_model_v2.pth`.

### Building a Graph from a TIR Program

```python
from tir_graph_builder_v2 import GraphBuilder

builder = GraphBuilder(feat_dim=16)
graph = builder.build(ir_mod)  # ir_mod is a tvm.IRModule
```

## Model Overview

The graph has three node types: For (loops), Block (computation), and Buffer (data). Edges encode loop nesting (structural) and buffer read/write relations (dataflow). A separate 32-dimensional global feature vector captures GPU launch configuration, thread bindings, vectorisation, and unroll settings, since these tend to get diluted by mean pooling.

The architecture is: node projection → 3 GraphSAGE layers with residual connections → global mean pooling → concatenation with global feature branch → MLP regression head. The training loss combines Smooth L1 regression and a within-workload pairwise ranking loss.

## Hardware

All experiments were run on an NVIDIA RTX 4060 Laptop GPU with TVM 0.23.dev0, PyTorch 2.5.1, and PyTorch Geometric.

## Known Issues

- Only tested on matmul and convolution workloads. Other operators (attention, pooling, etc.) have not been evaluated.
- Single GPU platform only — results may differ on other hardware.
- GNN inference speed during online tuning has not been benchmarked against XGBoost.
- Graph-level features are manually designed. A learned pooling mechanism could potentially replace them.
