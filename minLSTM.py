# https://arxiv.org/abs/2410.01201v2
# INITIAL SOURCE: https://github.com/zxdclyz/minLSTM_pytorch/blob/main/minLSTM.py
import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module
from typing import Optional, Tuple, Union


# appendix B.1
def parallel_scan_log(
    log_coeffs: torch.Tensor, log_values: torch.Tensor
) -> torch.Tensor:
    a_star = torch.cumsum(log_coeffs, dim=1)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


# appendix B.3
def g(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# log-space version, appendix B.3.2
class minLSTM(Module):
    def __init__(self, input_size: int, expansion_factor: float = 1.0, reduce_to_input: bool = True,
                 bidirectional: bool = False) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        hidden_size = int(input_size * expansion_factor)

        # Forward direction
        self.to_hidden_and_gates_forward = Linear(input_size, hidden_size * 3, bias=False)

        # Backward direction (if bidirectional)
        if self.bidirectional:
            self.to_hidden_and_gates_backward = Linear(input_size, hidden_size * 3, bias=False)

        # Output projection
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.to_out = (
            Linear(output_size, input_size, bias=False)
            if reduce_to_input and (expansion_factor != 1.0 or bidirectional)
            else Identity()
        )

    def _forward_pass(self, x: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        f_gate, i_gate, hidden = self.to_hidden_and_gates_forward(x).chunk(3, dim=-1)

        if seq_len == 1:
            # Sequential Mode
            h_prev = prev_hidden.squeeze(1) if prev_hidden is not None else None
            f_t = torch.sigmoid(f_gate)
            i_t = torch.sigmoid(i_gate)
            tilde_h_t = g(hidden)
            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)
            if h_prev is not None:
                h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t
            else:
                h_t = i_prime_t * tilde_h_t
            out = h_t
        else:
            # Parallel Mode
            diff = F.softplus(-f_gate) - F.softplus(-i_gate)
            log_f = -F.softplus(diff)
            log_i = -F.softplus(-diff)
            log_h_0 = log_g(prev_hidden) if prev_hidden is not None else None
            log_tilde_h = log_g(hidden)
            if log_h_0 is not None:
                log_values = torch.cat([log_h_0, log_i + log_tilde_h], dim=1)
                log_coeffs = F.pad(log_f, (0, 0, 1, 0))
            else:
                log_values = log_i + log_tilde_h
                log_coeffs = log_f
            h_t = parallel_scan_log(log_coeffs, log_values)
            out = h_t[:, -seq_len:]

        return out

    def forward(
            self,
            x: torch.Tensor,
            prev_hidden: Optional[torch.Tensor] = None,
            return_next_prev_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = x.size(1)

        # Forward pass
        forward_out = self._forward_pass(x, prev_hidden)

        if self.bidirectional:
            # Backward pass
            x_reversed = x.flip(dims=[1])
            backward_out = self._forward_pass(x_reversed, prev_hidden)
            backward_out = backward_out.flip(dims=[1])

            # Combine forward and backward outputs
            out = torch.cat([forward_out, backward_out], dim=-1)
        else:
            out = forward_out

        next_prev_hidden = out[:, -1:] if seq_len > 1 else out
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out
        return out, next_prev_hidden