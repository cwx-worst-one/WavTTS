


class HuggingfaceHubertFrontend():
    def __init__(self, sr=16000, max_duration=):
        self.sr = sr 

    def process(self, sample, meta):
        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        if len(wav.shape) >= 2:
            wav = wav[0]

        
