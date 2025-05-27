# MGBR
Pytorch implementation for "A Multi-Interaction Graph Attention Network for Bundle Recommendation"

### Environment

- OS: Ubuntu 18.04 or higher version
- python == 3.8.11 or above
- supported(tested) CUDA versions: 12.4
- Pytorch == 1.9.0 or above

### Run the code
To train MGBR on dataset Youshu with GPU 1, simply run:

    python main.py
You can indicate GPU id or dataset with cmd line arguments, and the hyper-parameters are recorded in config.py. 