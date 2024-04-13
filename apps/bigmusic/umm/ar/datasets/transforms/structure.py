import torch
from torchaudio.functional import loudness

from apps.bigmusic.umm.ar.datasets.transforms.audio import to_energy


class ChorusDetectionTransform:
    def __init__(
        self,
        audio_key="audio",
        sample_rate=24000,
        min_duration=5,
        volume_quantile=0.7,
        contrast_db=3,
        left_ctx=5,
        right_ctx=5,
        granularity_in_secs=0.5,
        debug=False,
    ):
        self.audio_key = audio_key
        self.sample_rate = sample_rate
        self.granularity_in_secs = granularity_in_secs
        self.window_size = round(sample_rate * granularity_in_secs)
        self.min_duration = round(min_duration / granularity_in_secs)
        self.volume_quantile = volume_quantile
        self.contrast_db = contrast_db
        self.left_ctx = round(left_ctx / granularity_in_secs)
        self.right_ctx = round(right_ctx / granularity_in_secs)
        self.debug = debug

    def is_loud(self, volume, threshold, st, en):
        return volume[st:en].nanmean() >= threshold

    def is_onset(self, volume, st):
        if st == 0:
            return True, 0
        onset_volume = volume[st : st + self.left_ctx].nanmean()
        left_volume = volume[st - self.left_ctx : st].nanmean()
        diff = onset_volume - left_volume
        return diff >= self.contrast_db, diff

    def is_offset(self, volume, en):
        if en == len(volume):
            return True, 0
        offset_volume = volume[en - self.right_ctx : en].nanmean()
        right_volume = volume[en : en + self.right_ctx].nanmean()
        diff = offset_volume - right_volume
        return diff >= self.contrast_db, diff

    def index_to_time(self, index):
        return index * self.granularity_in_secs

    def find_key_points(self, volume, mode, start, end):
        if mode == "onset":
            match_fn = self.is_onset
            # For onset, favor the points with higher contrast
            sort_fn = lambda cands: map(
                lambda x: x[0], sorted(cands, key=lambda x: -x[1])
            )
        elif mode == "offset":
            match_fn = self.is_offset
            # For offset, favor the later points
            sort_fn = lambda cands: map(lambda x: x[0], reversed(cands))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        key_points = []
        cands = []
        for x in range(start, end):
            matched, diff = match_fn(volume, x)
            if not matched:
                continue
            if len(cands) == 0 or cands[-1][0] == x - 1:
                cands.append((x, diff))
            else:
                key_points.append(list(sort_fn(cands)))
                cands = [(x, diff)]
        if len(cands) > 0:
            key_points.append(list(sort_fn(cands)))
        return key_points

    def find_segment(self, onset_group, offset_group, volume, threshold):
        for onset in onset_group:
            for offset in offset_group:
                if offset - onset < self.min_duration:
                    continue
                if onset == 0 and offset == len(volume):
                    continue
                if self.is_loud(volume, threshold, onset, offset):
                    segment = (
                        "chorus",
                        self.index_to_time(onset),
                        self.index_to_time(offset),
                    )
                    return segment, onset, offset
        return None, None, None

    def find_chorus(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        audio = audio.float().cpu()
        volume = loudness(
            audio.unfold(-1, self.window_size, self.window_size).unsqueeze(1),
            sample_rate=self.sample_rate,
        )
        threshold = torch.nanquantile(volume, self.volume_quantile)
        # Find onsets and offsets
        curr_st = 0
        onsets = self.find_key_points(volume, "onset", 0, len(volume))
        offsets = self.find_key_points(
            volume, "offset", self.min_duration, len(volume) + 1
        )
        if len(onsets) == 0 or len(offsets) == 0:
            return []
        # Find segments
        segments = []
        for onset_group in onsets:
            onset_group = [x for x in onset_group if x >= curr_st]
            if len(onset_group) == 0:
                continue
            for offset_group in offsets:
                segment, onset, offset = self.find_segment(
                    onset_group, offset_group, volume, threshold
                )
                if segment is not None:
                    segments.append(segment)
                    curr_st = offset
                    break
        if self.debug:
            print(f"onsets: {onsets}, offsets: {offsets}, segments: {segments}")
        return segments

    def __call__(self, item):
        item["structure"] = self.find_chorus(item[self.audio_key])
        return item


class IntensityTransform:
    def __init__(
        self,
        audio_key="audio",
        sample_rate=24000,
        calculation_mode="max",
        intensity_hz=1,
        debug=False,
    ):
        self.audio_key = audio_key
        self.window_size = sample_rate // intensity_hz
        self.calculation_mode = calculation_mode
        self.debug = debug

    def get_intensity(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        return to_energy(
            audio.float(),
            window_size=self.window_size,
            calculation_mode=self.calculation_mode,
        )[0]

    def __call__(self, item):
        if self.audio_key not in item:
            raise ValueError(
                f"Intensity transform needs reference audio at {self.audio_key}"
            )
        item["intensity"] = self.get_intensity(item[self.audio_key])
        return item
