# pRRTC: GPU-Parallel RRT-Connect

[![arXiv VAMP](https://img.shields.io/badge/arXiv-2503.06757-b31b1b.svg)](https://arxiv.org/abs/2503.06757)

This repository holds the code for [IROS 2025 submission "pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning."](https://arxiv.org/abs/2503.06757)

We introduce pRRTC, a GPU-based, parallel RRT-Connect-based algorithm. Our approach has three key improvements: 
- Concurrent sampling, expansion and connection of start and goal trees via GPU multithreading
- SIMT-optimized collision checking to quickly validate edges, inspired by the SIMD-optimized validation of [Vector-Accelerated Motion Planning](https://github.com/KavrakiLab/vamp/tree/main)
- Efficient memory management between block- and thread-level parallelism, reducing expensive memory transfer overheads

Our empirical evaluations show that pRRTC achieves up to 6x average speedup on constrained reaching tasks at high collision checking resolution. pRRTC also demonstrates a 5x reduction in solution time variance and 1.5x improvement in initial path costs compared to state-of-the-art motion planners in complex environments.

## Supported Robots
pRRTC currently supports 7-DoF Franka Emika Panda, 8-DoF Fetch, and 14-DoF Rethink Robotics Baxter.

## Building Code
To build pRRTC, follow the instructions below
```
git clone git@github.com:CoMMALab/pRRTC.git
cmake -B build
cmake --build build
```

## Running Code
The repository comes with two benchmarking scripts: evaluate_mbm.cpp and single_mbm.cpp.

evaluate_mbm.cpp allows users to benchmark pRRTC's performance using Panda, Fetch, or Baxter on the entire [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) dataset. To run evaluate_mbm.cpp:
```
build/evaluate_mbm <robot> <experiment name>
```

single_mbm.cpp allows users to benchmark pRRTC's performance using Panda, Fetch, or Baxter on a single problem of [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker). To run single_mbm.cpp:
```
build/single_mbm <robot> <MBM problem name> <MBM problem index>
```

## Planner Configuration
pRRTC has the following parameters which can be modified in the benchmarking scripts:
- <ins>**max_samples**</ins>: maximum number of samples in trees
- <ins>**max_iters**</ins>: maximum number of planning iterations
- <ins>**num_new_configs**</ins>: amount of new samples generated per iteration
- <ins>**range**</ins>: maximum RRT-Connect extension range
- <ins>**granularity**</ins>: number of discretized motions along an edge during collision checking
- <ins>**balance**</ins>: whether to enable tree balancing -- 0 for no balancing; 1 for distributed balancing where each iteration may generate samples for one or two trees; 2 for single-sided balancing where each iteration generate samples for one tree only
- <ins>**tree_ratio**</ins>: the threshold for distinguishing which tree is smaller in size -- if balance set to 1, then set tree_ratio to 0.5; if balance set to 2, then set tree_ratio to 1
- <ins>**dynamic_domain**</ins>: whether to enable [dynamic domain sampling](https://ieeexplore.ieee.org/abstract/document/1570709) -- 0 for false; 1 for true
- <ins>**dd_alpha**</ins>: extent to which each radius is enlarged or shrunk per modification
- <ins>**dd_radius**</ins>: starting radius for dynamic domain sampling
- <ins>**dd_min_radius**</ins>: minimum radius for dynamic domain sampling




