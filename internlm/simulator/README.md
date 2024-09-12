# InternLM Simulator


## 1. Introduction
The solver mainly consists of two components:
1. `profiling`: Collects the time consumption of each stage during the model training process in advance and saves it as data files and image files.
2. `simulation`: Simulates the model training process based on the collected data files and outputs the time consumption of each stage during the training process.

## 2. Usage

### 2.1 Generate profiling data

There are two types of profiling data:
1. '`linear`' profiling data, include: [`LINEAR`]
2. '`Communication`' profiling data, include: [`ALL2ALL`, `ALLREDUCE`, `REDUCESCATTER`, `ALLGATHER`, `BROADCAST`]


Note:
1. It is recommended to use more than 64 GPUs for data collection to ensure more accurate communication data.
2. `Flash Attention` information is not collected in advance but is collected on the fly during the simulation and stored in the cache. This is because there are many variables that affect the performance of flash attention, and collecting in advance cannot cover all variables.

```python
# generate profiling data
torchrun --nproc-per-node=8  gen_profiler_data.py

# the profiling data will be saved in the following path
./prof_data
├── data.pt
└── pics
    ├── cal
    │   └── linear.jpg
    └── comm
        ├── all2all_intra_2_inter_1.jpg
        ├── all2all_intra_4_inter_1.jpg
        ├── all_gather_intra_2_inter_1.jpg
        ├── all_gather_intra_4_inter_1.jpg
        ├── all_reduce_intra_2_inter_1.jpg
        ├── all_reduce_intra_4_inter_1.jpg
        ├── broadcast_intra_2_inter_1.jpg
        ├── broadcast_intra_4_inter_1.jpg
        ├── reduce_scatter_intra_2_inter_1.jpg
        └── reduce_scatter_intra_4_inter_1.jpg

```

### 2.2 Run simulation
Running the solver does not require a GPU (although some packages may require a GPU environment, if you encounter any issues, please raise an issue). Currently, the solver only supports the formulaic solving method using simulation_train_formulaic.py, which requires a config file and profiling data file as follows:

```bash

python simulation_train_formulaic.py --pre_profiling_data_path ./prof_data/data.pt --config configs/7B_internlm2.py --run_all_solu --model_size 7 --world_size 128 --global_batch_size 4194304

# explanation:
python simulation_train_formulaic.py
    --pre_profiling_data_path ./prof_data/data.pt    # profiling data file
    --config configs/7B_internlm2.py                 # model configuration file
    --run_all_solu                                   # whether to iterate and solve all possible solutions
    --model_size 7                                   # means 7B model, if you want to run 70B model, you can set model_size to 70
    --world_size 128                                 # solving range is 128 cards
    --global_batch_size 4194304                      # global batch size, 4M
```
