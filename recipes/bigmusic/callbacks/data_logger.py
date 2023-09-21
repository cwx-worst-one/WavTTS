from typing import Any, Dict, List, Union

import torch

from samantha.callbacks.data_logger import AudioDataLogger, DataUsageLogger


class BigMusicDataUsageLogger(DataUsageLogger):
    
    def get_batch_data(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        data = {}
        # data["style_text"] = batch["style_text"]
        # data["normalized_text"] = batch["normalized_text"]
        # data["lyrics_tokens"] = batch["lyrics_tokens"]
        # data["speaker_id"] = batch["speaker_id"]
        # data["style_metadata"] = batch["style_metadata"]
        data["song_id"] = batch["song_id"]
        data["shard"] = batch["shard"]
        data["worker_id"] = batch["worker_id"]
        return data


class BigMusicAudioDataLogger(AudioDataLogger):
    def get_audios_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["target_audio"]
