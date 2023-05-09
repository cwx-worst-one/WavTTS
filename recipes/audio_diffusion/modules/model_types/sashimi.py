import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import SigmaEmbedding
from .s4_layer.s4 import S4, LinearActivation

# Wrapper adding sigma conditioning to SaShiMi model


class SashimiDiffWrapper(nn.Module):
    def __init__(
        self, width=32, num_signal_channels=2, sigma_embedding_dim=32, **kwargs
    ):
        super().__init__()
        self.sashimi_model = Sashimi(unet=True, d_model=width, **kwargs)
        self.input_mapping = nn.Conv1d(num_signal_channels, width, 1)
        self.output_mapping = nn.Conv1d(width, num_signal_channels, 1)
        self.sigma_embedding = SigmaEmbedding(
            embedding_dim=sigma_embedding_dim, output_dim=width
        )

    def forward(self, x, sigma):
        embedded_sigma = self.sigma_embedding(sigma)
        x = self.input_mapping(x) + embedded_sigma
        x, _ = self.sashimi_model(x)
        return self.output_mapping(x)


class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(d_input * pool, self.d_output, transposed=True)

    def forward(self, x):
        x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None:
            return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), "... h s -> ... (h s)")
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool

        self.linear = LinearActivation(d_input, self.d_output * pool, transposed=True)

    def forward(self, x, skip=None):
        x = self.linear(x)

        x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, "... (h s) l -> ... h (l s)", s=self.pool)

        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, "... (h s) -> ... h s", s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else:
            assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(
            batch_shape + (self.d_output, self.pool), device=device
        )  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state


class FFBlock(nn.Module):
    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.
        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = LinearActivation(
            d_model, d_model * expand, transposed=True, activation="gelu", activate=True
        )
        dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand, d_model, transposed=True, activation=None, activate=False
        )

        self.ff = nn.Sequential(input_linear, dropout, output_linear)

    def forward(self, x):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        return self.ff(x.unsqueeze(-1)).squeeze(-1), state


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer, dropout=0.0):
        """
        Residual S4 block.
        Args:
            d_model: dimension of the model
            layer: a layer config
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inp):
        """
        Input x is shape (B, d_input, L)
        """

        # Prenorm
        z = self.norm(inp.transpose(-1, -2)).transpose(-1, -2)

        # Apply layer: we ignore the state input and output for training
        z_out, _ = self.layer(z)

        # Dropout on the output of the layer
        z_drop = self.dropout(z_out)

        # Residual connection
        x = z_drop + inp

        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)

        # Residual connection
        x = z + x

        return x, state


class Sashimi(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_layers=8,
        pool=[4, 4],
        expand=2,
        ff=2,
        bidirectional=False,
        unet=False,
        dropout=0.0,
        **s4_args,
    ):
        """
        SaShiMi model backbone.
        Args:
            d_model:
                dimension of the model. default 65
            n_layers:
                number of (Residual (S4) --> Residual (FF)) blocks
                at each pooling level.
            pool:
                pooling factor at each level. Pooling shrinks the
                sequence length at lower levels.
            expand:
                channel expansion factor when pooling.
            ff:
                expansion factor for the FF inverted bottleneck.
            bidirectional:
                use bidirectional S4 layers.
            unet:
                use a unet-like architecture
            dropout:
                dropout rate. Default to 0.0.
        """
        super().__init__()
        self.d_model = H = d_model
        self.d_output = H
        self.unet = unet

        def s4_block(dim):
            layer = S4(
                d_model=dim,
                d_state=64,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=True,
                **s4_args,
            )
            return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

        def ff_block(dim):
            layer = FFBlock(d_model=dim, expand=ff, dropout=dropout)
            return ResidualBlock(d_model=dim, layer=layer, dropout=dropout)

        # Down blocks
        d_layers = []
        for p in pool:
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    d_layers.append(s4_block(H))
                    if ff > 0:
                        d_layers.append(ff_block(H))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand

        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H))
            if ff > 0:
                c_layers.append(ff_block(H))

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p))

            for _ in range(n_layers):
                block.append(s4_block(H))
                if ff > 0:
                    block.append(ff_block(H))

            u_layers.append(nn.ModuleList(block))

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        assert H == d_model

    def forward(self, inp, state=None):
        x = self.norm(inp.transpose(-1, -2)).transpose(-1, -2)
        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            x, _ = layer(x)
            outputs.append(x)

        # Center block
        for layer in self.c_layers:
            x, _ = layer(x)
        x = (
            x + outputs.pop()
        )  # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    x, _ = layer(x)
                    x = x + outputs.pop()  # skip connection
            else:
                for layer in block:
                    x, _ = layer(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()  # add a skip connection from the input

        # feature projection
        out = self.norm(x.transpose(-1, -2)).transpose(-1, -2)

        return out, None  # required to return a state

    def default_state(self, *args, **kwargs):
        layers = (
            list(self.d_layers)
            + list(self.c_layers)
            + [layer for block in self.u_layers for layer in block]
        )
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = []  # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None:
                break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped // 3 :]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()

        # feature projection
        # x = self.norm(x)
        return x, next_state

    def setup_rnn(self, mode="dense"):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.
        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%
                `linear` should be faster theoretically but is slow in practice since it
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the
                diagonalization is numerically unstable(although we haven't encountered
                this case in practice), while `dense` should always be stable.
        """
        assert mode in ["dense", "diagonal", "linear"]
        for module in self.modules():
            if hasattr(module, "setup_step"):
                module.setup_step(mode=mode)
