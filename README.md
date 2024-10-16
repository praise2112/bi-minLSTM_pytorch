# minLSTM_pytorch
Bi-directional minLSTM in PyTorch

> Adapted from [minGRU_pytorch](https://github.com/lucidrains/minGRU-pytorch) and [minLSTM](https://github.com/zxdclyz/minLSTM_pytorch/tree/main)

A numerically stable log-space version of minLSTM in PyTorch, as proposed in [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201v2)

## Usage

```python
import torch
from minLSTM import minLSTM

# with bi-directionality
model = minLSTM(input_size=128, expansion_factor=1.5, bidirectional=True)


x = torch.randn(1, 1024, 128)

# Parallel mode when seq_len > 1
parallel_out = model(x)[:, -1:]

# Sequential mode when seq_len = 1
prev_hidden = None
for x_t in x.unbind(dim=1):
    sequential_out, prev_hidden = model(
        x_t[:, None, :], prev_hidden, return_next_prev_hidden=True
    )

assert torch.allclose(parallel_out, sequential_out, atol=1e-4)
```