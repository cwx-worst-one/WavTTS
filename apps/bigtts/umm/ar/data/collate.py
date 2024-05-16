import torch
from typing import List, Dict, Any
from samantha.dataio.lite.transform import CollatorBase
from samantha.transforms.audio import RandomPad, Pad


class ARCollator(CollatorBase):
    def __init__(
        self,
        audio_key: str = "wav",
        token_key: str = "umm_token",
        split_by_alignment: bool = False,
    ):
        super().__init__()
        self.audio_key = audio_key
        self.token_key = token_key
        self.split_by_alignment = split_by_alignment

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) > 0 and all("audio" in x for x in batch):
            return self.collate_audio(batch)
        return self.collate_token(batch)

    def collate_audio(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max([x.get(self.audio_key).shape[-1] for x in batch])
        random_pad = RandomPad(n_samples=max_length)

        audio, token = [], []
        for x in batch:
            audio.append(random_pad(x.get(self.audio_key)))
            token.append(x.get("token", torch.zeros(0).long()))

        return {
            "target_audio": torch.stack(audio, dim=0),
            "conditions": "lyrics_tokens",
            "lyrics_tokens": torch.nn.utils.rnn.pad_sequence(
                token, batch_first=True, padding_value=0
            ),
        }

    def collate_token(self, batch: List[Dict[str, torch.Tensor]]) -> Dict:
        max_length = max([x.get(self.token_key).shape[-1] for x in batch])
        zero_pad = Pad(n_samples=max_length)

        umm_token, umm_token_length = [], []
        token, phone, tone, wordseg, lang = [], [], [], [], []
        for x in batch:
            umm_token.append(zero_pad(x.get(self.token_key).unsqueeze(0)))
            umm_token_length.append(x.get(self.token_key).numel())

            token.append(x.get("token", torch.zeros(0).long()))
            phone.append(x.get("phone", torch.zeros(0).long()))
            tone.append(x.get("tone", torch.zeros(0).long()))
            wordseg.append(x.get("wordseg", torch.zeros(0).long()))
            lang.append(x["lang"])

        res = {
            "target_ids": torch.cat(umm_token, dim=0),
            "target_ids_length": torch.tensor(umm_token_length),
            "lyrics_token_length": torch.tensor([x.numel() for x in token]),
            "conditions": "lyrics_tokens",
            "lyrics_tokens": torch.nn.utils.rnn.pad_sequence(
                token, batch_first=True, padding_value=0
            ),
            "phones": torch.nn.utils.rnn.pad_sequence(
                phone, batch_first=True, padding_value=0
            ),
            "tones": torch.nn.utils.rnn.pad_sequence(
                tone, batch_first=True, padding_value=0
            ),
            "wordsegs": torch.nn.utils.rnn.pad_sequence(
                wordseg, batch_first=True, padding_value=0
            ),
            "lang": torch.tensor(lang),
        }

        if self.split_by_alignment:
            res.update(self._split_by_alignment(batch))
        return res

    def _split_by_alignment(self, batch):
        max_length = max([x.get("prompt_" + self.token_key).shape[-1] for x in batch])
        zero_pad = Pad(n_samples=max_length)
        prompt_umm_token = []
        prompt_umm_token_length = []
        for item in batch:
            prompt_umm_token.append(
                zero_pad(item.get("prompt_" + self.token_key).unsqueeze(0))
            )
            prompt_umm_token_length.append(item.get("prompt_" + self.token_key).numel())
        return {
            "prompt_ids": torch.cat(prompt_umm_token, dim=0),
            "prompt_ids_length": torch.tensor(prompt_umm_token_length),
        }
