import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.utilities import grad_norm
from tqdm import tqdm

from recipes.soundstorm.lightning.masking_scheme import MaskingScheme, cosine_schedule, DucMaskingScheme, SoundStormMaskingScheme
from samantha.models.conformer import Conformer, ConformerConfig  # noqa
from samantha.models.llama import LlamaConfig, LlamaModel
from samantha.components.attention import MultiHeadAttention
from dataclasses import dataclass, field

def upsample_tokens(token_ids: torch.Tensor, rate: int) -> torch.Tensor:
    """This function duplicates token_ids of shape [batch, seq_len] by a given rate,
    repeating them N (rate) times after each other.

    Args:
        token_ids (torch.Tensor): _description_
        rate (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    return rearrange(token_ids.unsqueeze(dim=1).repeat(1, rate, 1), "b r s -> b (s r)")



@dataclass
class SoundStormConfig:
    sample_rate: int
    n_embd: int
    n_head: int
    n_layer: int
    masking_scheme: MaskingScheme = SoundStormMaskingScheme(
        sample_q_uniformly=False,
        sample_t=True
    )
    fine_quantizer_embedding_dropout: bool = True
    conditioning_dropout: Optional[float] = None
    frontend: str = "sum"
    frontend_n_head: int = 4
    frontend_rope: bool = True
    attention_kwargs: Optional[dict] = field(default_factory=dict)

class SoundStorm(pl.LightningModule):
    """At the input side, we interleave the time-aligned conditioning tokens
    with the SoundStream tokens at the frame level, embed the resulting
    sequence, sum the embeddings corresponding to the same frame, including
    the embedding of the conditioning token, and pass the resulting continuous
    embeddings to a Conformer. Consequently, the sequence length for bidirectional
    self-attention in the Conformer is determined by the number of SoundStream
    frames (typically 50 per second), and thus is independent of the number of
    RVQ levels Q, allowing one to handle audio with length on the order of minutes.
    At the output side, we use Q dense layers as heads to produce the target
    SoundStream tokens."""

    def __init__(
        self,
        config: SoundStormConfig,
        audio_model: pl.LightningModule,
        optimizer_class,
        scheduler_class,
        semantic_model: Optional[pl.LightningModule] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["audio_model", "semantic_model"])
        self.config = config
        self.audio_model = audio_model.eval()
        self.audio_model.freeze()

        # self.semantic_model = SemanticModel().eval()
        # self.semantic_model.freeze()
        self.semantic_model = semantic_model

        if self.semantic_model is not None:
            self.semantic_to_audio_rate = (
                self.audio_model.frame_rate // self.semantic_model.frame_rate
            )
            self.semantic_embedding = nn.Embedding(
                self.semantic_model.codebook_size, config.n_embd
            )
            self.semantic_uncond_token_id = self.semantic_model.codebook_size

        self.n_quantizers = self.audio_model.n_quantizers
        self.out_dim = self.audio_model.codebook_size
        self.mask_token_id = self.audio_model.codebook_size

        self.transformer_config = LlamaConfig(
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            use_rotary_embeddings=True,
            causal=False,
            attention_kwargs=config.attention_kwargs,
        )
        self.audio_embedding = nn.ModuleList(
            [
                nn.Embedding(self.audio_model.codebook_size + 1, config.n_embd)
                for _ in range(self.n_quantizers)
            ]
        )

        self.transformer = LlamaModel(self.transformer_config)
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, self.out_dim, bias=False)
                for _ in range(self.n_quantizers)
            ]
        )

        if config.frontend == "mha":
            self.mha_frontend = MultiHeadAttention(
                d_model=config.n_embd,
                n_heads=config.frontend_n_head,
                bias=False,
                use_rotary_embeddings=config.frontend_rope,
                max_seq_len=self.n_quantizers + 1,
            )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Reinitialize selected weights subject to the OpenAI GPT-2 Paper
        Scheme: A modified initialization which accounts for the accumulation
        on the residual path with model depth. Scale the weights of residual
        layers at initialization by a factor of 1/√N where N is the # of
        residual layers.

        Args:
            module (_type_): _description_
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.transformer_config.initializer_range / math.sqrt(2 * self.transformer_config.n_layer),
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.transformer_config.initializer_range / math.sqrt(2 * self.transformer_config.n_layer),
            )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2), sync_dist=True)

    def prepare_audio_embedding(
        self, audio_tokens: torch.Tensor, quantizers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        audio_embs = []
        for q in range(self.n_quantizers):
            audio_embs.append(self.audio_embedding[q](audio_tokens[:, q]))
        audio_embs = torch.stack(audio_embs, dim=2)

        if self.config.fine_quantizer_embedding_dropout:
            for idx in range(audio_embs.shape[0]):
                audio_embs[idx, :, quantizers[idx] + 1 :] = 0
        return audio_embs

    def prepare_semantic_embedding(self, semantic_tokens: Optional[torch.Tensor] = None):
        if semantic_tokens is None:
            return None
        
        if self.training and self.config.conditioning_dropout is not None:
            batch_dropout = (
                torch.rand(semantic_tokens.shape[0]) < self.config.conditioning_dropout
            )
            semantic_tokens[batch_dropout] = self.semantic_uncond_token_id

        semantic_tokens = upsample_tokens(semantic_tokens, self.semantic_to_audio_rate)
        semantic_emb = self.semantic_embedding(semantic_tokens).unsqueeze(dim=2)    # (B, T, 1, D)
        return semantic_emb


    def prepare_inputs(self, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        audio = batch["audio"]
        with torch.no_grad():
            audio = audio.mean(dim=1, keepdim=True)
            audio_tokens = self.audio_model(audio)
            semantic_tokens = None
            # semantic_tokens = self.semantic_model(audio) if self.semantic_model is not None else None
        return {
            "semantic_tokens": semantic_tokens,
            "audio_tokens": audio_tokens,
            **batch
        }

    def forward(
        self,
        audio_tokens: torch.Tensor,
        selected_qs: torch.Tensor,
        semantic_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model, given a batch of audio.

        NOTE: Returns a tensor of logits of the quantizer heads listed in `selected_qs`,
        so the quantizer dimension may not line up linearly!

        At the input side, we interleave the time-aligned conditioning tokens
        with the SoundStream tokens at the frame level, embed the resulting
        sequence, sum the embeddings corresponding to the same frame, including
        the embedding of the conditioning token, and pass the resulting
        continuous embeddings to a Conformer. Consequently, the sequence length
        for bidirectional self-attention in the Conformer is determined by the
        number of SoundStream frames (typically 50 per second), and thus is
        independent of the number of RVQ levels $Q$, allowing one to handle audio
        with length on the order of minutes. At the output side, we use $Q$ dense
        layers as heads to produce the target SoundStream tokens

        We embed the resulting sequence, sum the embeddings corresponding to the
        same frame, including the embedding of the conditioning token, and pass
        the resulting continuous embeddings to a Conformer

        Args:
            semantic_tokens (torch.Tensor): _description_
            audio_tokens (torch.Tensor): _description_
            sampled_q (Optional[int], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        semantic_emb = self.prepare_semantic_embedding(semantic_tokens)
        audio_emb = self.prepare_audio_embedding(audio_tokens, selected_qs)   # (B, T, Q, D)
        B, T, _, D = audio_emb.shape

        if self.config.frontend == "sum":
            if semantic_tokens is not None:
                feats = torch.cat((semantic_emb, audio_emb), dim=2).sum(dim=2) # (B, T, D)
            else:
                feats = audio_emb.sum(dim=2) # (B, T, D)

        elif self.config.frontend == "mha":
            query = [audio_emb[i, :, selected_qs[i]] for i in range(B)]
            query = torch.stack(query, dim=0).reshape(B * T, -1, D) # (B * T, 1, D)

            if semantic_tokens is not None:
                feats = torch.cat((semantic_emb, audio_emb), dim=2)    # (B, T, 1 + Q, D)
            else:
                feats = audio_emb    # (B, T, 1 + Q, D)

            feats = feats.reshape(B * T, -1, D) # (B * T, 1 + Q, D)
            feats = self.mha_frontend(x=query, context=feats)   # (B * T, 1, D)
            feats = feats.reshape(B, T, -1) # (B, T, D)
        else:
            raise ValueError(f"Unknown frontend: {self.config.frontend}")

        x = self.transformer(feats)
        logits = []
        for batch_idx in range(B):
            logits.append(self.heads[selected_qs[batch_idx]](x[batch_idx]))
        return torch.stack(logits, dim=0)


    def step(self, batch: Dict[str, torch.Tensor], return_loss: bool = True) -> torch.Tensor:
        """
        Args:
            batch (Dict[str, torch.Tensor]): Audio data
            return_loss (bool, optional): Whether to return the loss value or the
                predicted logits.

        Returns:
            torch.Tensor: Loss value or the predicted logits
        """
        inputs = self.prepare_inputs(batch)
        masked_audio_tokens, _, rand_qs = self.config.masking_scheme(
            inputs["audio_tokens"], self.mask_token_id
        )
        logits = self.forward(masked_audio_tokens, rand_qs, semantic_tokens=inputs["semantic_tokens"])

        if return_loss:
            return self.loss(logits, inputs["audio_tokens"], masked_audio_tokens, rand_qs)
        return logits

    def loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masked_audio_tokens: torch.Tensor,
        selected_qs: torch.Tensor,
        report_q_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute the cross-entropy loss (negative log-likelihood) on the predicted
        and target token id's. We only calculate the loss on the masked tokens (within the
        q-th RVQ level). Hence the shapes are:

        preds: [n_samples, audio_codebook_size]
        y = [n_samples] ($C \in audio_codebook_size$)

        Args:
            preds (torch.Tensor): _description_
            targets (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor]: Dictionary containing the cross entropy loss and
            perplexity
        """
        batch_indices = torch.arange(targets.shape[0], device=targets.device)
        targets = targets[batch_indices, selected_qs]
        mask = masked_audio_tokens[batch_indices, selected_qs] == self.mask_token_id

        masked_preds = preds[mask]
        masked_targets = targets[mask]
        accuracy = (masked_preds.argmax(1) == masked_targets).float().mean()
        loss = F.cross_entropy(masked_preds, masked_targets)

        if report_q_loss:
            q_losses = torch.zeros(self.n_quantizers, device=self.device)
            q_count = torch.zeros(self.n_quantizers, device=self.device)
            for batch_idx, q in enumerate(selected_qs):
                q_mask = mask[batch_idx]
                q_preds = preds[batch_idx, q_mask]
                q_targets = targets[batch_idx, q_mask]
                q_losses[q] += F.cross_entropy(q_preds, q_targets)
                q_count[q] += 1

            q_count[q_count == 0] = 1e-7
            q_losses = q_losses / q_count
            for q in range(self.n_quantizers):
                self.log(f"loss_q/{q}", q_losses[q])

        return {"loss": loss, "ppl": loss.exp(), "accuracy": accuracy}

    @torch.no_grad()
    def iterative_decoding(
        self,
        max_seq_len: int,
        iterations: List[int],
        score_strategies: List[str],
        semantic_tokens: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        temperatures: Optional[List[float]] = None,
        sampled_t: Optional[int] = None,
        seed_tokens: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """Iterative decoding scheme from the SoundStorm/MaskGIT papers.

        Given a conditioning signal, our decoding scheme starts with all SoundStream tokens masked out
        except for the ones of the prompt (if provided). Then, it proceeds to sampling the tokens RVQ
        level-wise in a coarse-to-fine order, only proceeding to level q + 1 when all tokens for levels
        1,...,q have been sampled. Within an RVQ level, we use the confidence-based sampling scheme
        of Chang et al. (2022). Namely, we perform multiple forward passes, and at each iteration i, we
        sample candidates for the masked positions, retaining p_i of them based on confidence scores, where
        p_i follows a cosine schedule. Compared to Chang et al. (2022), we use greedy decoding instead of
        confidence-based sampling for the last iteration within each RVQ level, which we found to improve
        the perceived audio quality.

        Args:
            semantic_tokens (torch.Tensor): _description_
            n_quantizers (int): _description_
            max_seq_len (int): _description_
            mask_token_id (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        batch_size = semantic_tokens.shape[0] if semantic_tokens is not None else seed_tokens.shape[0]
        audio_tokens = torch.full(
            (batch_size, self.n_quantizers, max_seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )

        if sampled_t:
            audio_tokens[..., :sampled_t] = seed_tokens[..., :sampled_t]

        start_quantizer = 0
        # TODO: refactor 
        # if seed_tokens is None:
        #     start_quantizer = 0
        # else:
        #     start_quantizer = seed_tokens.shape[1]
        #     audio_tokens[:, : seed_tokens.shape[1]] = seed_tokens

        if prefix_tokens is not None:
            ratio = 1 - float(prefix_tokens.shape[2]) / max_seq_len
            iterations = [max(int(i * ratio), 1) for i in iterations]
            audio_tokens[:, :, : prefix_tokens.shape[2]] = prefix_tokens

        if temperatures is None:
            temperatures = [1.0] * self.n_quantizers

        metrics = defaultdict(list)
        for q in tqdm(
            range(start_quantizer, self.n_quantizers),
            desc="Iteratively decoding audio tokens...",
        ):
            q_iter = iterations[q]
            q_score_strategy = score_strategies[q]
            q_temperature = temperatures[q]
            ratios = torch.linspace(0, 1.0, q_iter + 1)[1:]
            cos_ratios = cosine_schedule(ratios)
            quantizers = torch.LongTensor([q] * batch_size).to(self.device)
            for step_idx, ratio in enumerate(cos_ratios):
                logits = self.forward(audio_tokens, quantizers, semantic_tokens=semantic_tokens)

                if guidance_scale is not None:
                    uncond_semantic_tokens = torch.full_like(
                        semantic_tokens, self.semantic_uncond_token_id
                    )
                    uncond_logits = self.forward(
                        audio_tokens, quantizers, semantic_tokens=uncond_semantic_tokens
                    )
                    q_logits = (
                        guidance_scale * q_logits + (1 - guidance_scale) * uncond_logits
                    )

                probs = logits.softmax(dim=-1)
                masked_positions = audio_tokens[:, q] == self.mask_token_id

                if masked_positions.sum() == 0:
                    continue

                if step_idx == q_iter - 1:
                    # greedy decoding for the last iteration
                    sampled_tokens = probs.argmax(dim=-1)
                    audio_tokens[:, q] = torch.where(
                        masked_positions, sampled_tokens, audio_tokens[:, q]
                    )
                else:

                    # sample candidates first
                    probs_scaled = (logits / q_temperature).softmax(dim=-1)
                    sampled_tokens = torch.distributions.categorical.Categorical(
                        probs_scaled
                    ).sample()

                    # gather the probabilities of each of the candidates
                    if q_score_strategy == "maskgit":
                        scores = probs.gather(
                            2, rearrange(sampled_tokens, "b n -> b n 1")
                        )
                        scores = rearrange(scores, "b n 1 -> b n")
                    elif q_score_strategy == "max_prob":
                        scores, _ = probs.max(dim=-1)
                    elif q_score_strategy == "max_entropy":
                        scores = torch.distributions.categorical.Categorical(
                            probs
                        ).entropy()
                    elif q_score_strategy == "min_entropy":
                        scores = (
                            torch.distributions.categorical.Categorical(probs).entropy()
                            * -1
                        )
                    elif q_score_strategy == "random":
                        scores = torch.rand(probs.shape[:2]).to(probs.device)
                    elif q_score_strategy == "sequential":
                        m, n = probs.shape[:2]
                        scores = torch.arange(n).repeat(m, 1).to(probs.device) * -1.0
                    else:
                        raise ValueError(f"Unknown score strategy: {q_score_strategy}")

                    # keep only the top k scores
                    # we assume an equal unmasking schedule, so we simply take the number
                    # of masked positions of the first batch element as our reference
                    tokens_left = masked_positions[0].sum()
                    tokens_unmasked = masked_positions.shape[-1] - tokens_left
                    topk_tokens = ((1 - ratio) * masked_positions.shape[-1]).long()
                    topk_tokens = topk_tokens - tokens_unmasked

                    # always select at least 1
                    topk_tokens = max(topk_tokens, 1)
                    # select the topk, otherwise the remainder
                    topk_tokens = min(topk_tokens, tokens_left)


                    # don't select topk of previously sampled tokens
                    scores = torch.where(
                        masked_positions, scores, -torch.finfo(scores.dtype).max
                    )
                    # batched topk
                    topk_probs, topk_indices = scores.topk(topk_tokens, dim=-1)

                    # create a mask that is True for all scores that meet the confidence criterium
                    confidence_mask = torch.zeros_like(
                        scores, dtype=torch.bool
                    ).scatter(dim=-1, index=topk_indices, value=True)

                    # only fill positions that are currently masked and meet the confidence criterium
                    # otherwise fill with original token
                    fill_positions = masked_positions & confidence_mask
                    audio_tokens[:, q] = torch.where(
                        fill_positions, sampled_tokens, audio_tokens[:, q]
                    )

                    if debug:
                        metrics["audio_tokens"].append(audio_tokens.clone().cpu())
                        metrics["sampled_tokens"].append(sampled_tokens.clone().cpu())
                        metrics["scores"].append(scores.clone().cpu())
                        metrics["probs"].append(probs.clone().cpu())
                        metrics["topk_probs"].append(topk_probs.clone().cpu())
                        metrics["topk_indices"].append(topk_indices.clone().cpu())
                        metrics["tokens_left"].append(tokens_left)
        return audio_tokens, metrics


    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/train", v, rank_zero_only=True, prog_bar=True)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/valid", v, rank_zero_only=True)
        return loss["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, return_loss=False)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
