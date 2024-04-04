import torch
import torch.nn.functional as F

from samantha.criterion.criterion import MaskedCrossEntropy


class RefMaskedCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous().float()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return {"loss": loss.mean()}

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        # bug: original implement cal loss with batch mask.
        # loss = (loss / mask.sum()).sum()
        loss = torch.mean(
            loss.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)
        )
        return {"loss": loss}


def test_masked_cross_entropy():
    B, T, D = 8, 128, 1024
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logits = torch.randn([B, T, D], device=device)
    targets = torch.randint(0, D, (B, T), device=device)
    valid_lengths = torch.randint(0, T, (B,), device=device)
    mask = (
        valid_lengths[:, None]
        > torch.arange(T, dtype=torch.long, device=device)[None, :]
    )
    ref_masked_ce, masked_ce = RefMaskedCrossEntropy(), MaskedCrossEntropy()
    ref_loss_wo_mask = ref_masked_ce(logits, targets)["loss"]
    loss_wo_mask = masked_ce(logits, targets)
    torch.testing.assert_close(loss_wo_mask, ref_loss_wo_mask)

    ref_loss_w_mask = ref_masked_ce(logits, targets, mask)["loss"]
    loss_w_mask = masked_ce(logits, targets, mask)
    torch.testing.assert_close(loss_w_mask, ref_loss_w_mask)
