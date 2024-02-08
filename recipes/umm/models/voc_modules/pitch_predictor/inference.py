from recipes.umm.models.voc_modules.pitch_predictor.model import PitchPredictor


class PerceptualPitchPredictor:
    """
    Perceptual Pitch Predictor wrapper to standardize interface like RVMPE pitch model.
    @hanoihantrakul 1/25/2024
    """

    def __init__(self):
        self.wrapped_model = PitchPredictor()
        # This is required to prevent Pytorch Lightning from thinking this model is a trainable module.
        # Without this requires_grad = False parameter you will get a DDPStrategy error.
        for p in self.wrapped_model.parameters():
            p.requires_grad = False
        # All perceptual losses in UMMMv2 are standardized to mel 160
        self.n_mels_in = 160

    def load_and_eval(self, state_dict):
        try:
            self.wrapped_model.load_state_dict(state_dict)
            self.wrapped_model.eval()
            print("Succesfully loaded Perceptual Pitch Predictor!")
        except Exception as e:
            print(e)

    def forward(self, x):
        assert x.shape[-1] == self.n_mels_in
        self.wrapped_model.to(x.device)
        return self.wrapped_model.forward(x)

    def get_hidden_state(self):
        return self.wrapped_model.get_hidden_state()
