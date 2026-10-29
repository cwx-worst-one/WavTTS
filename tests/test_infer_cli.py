import inspect
import runpy
import sys

import pytest

from wavtts.infer import utils_infer


@pytest.mark.parametrize("name", ["cfg_strength", "cross_fade_duration", "sway_sampling_coef"])
@pytest.mark.parametrize("value", [None, 0.0, 0.5])
def test_cli_sampling_overrides(tmp_path, monkeypatch, name, value):
    config = tmp_path / "inference.toml"
    config.write_text(f"{name} = 0.75\n", encoding="utf-8")
    argv = ["wavtts_infer-cli", "-c", str(config), "--ckpt_file", "local.pt"]
    if value is not None:
        argv.extend([f"--{name}", str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    resolved = {}

    class ModelLoadReached(Exception):
        pass

    def stop_before_model_load(*args, **kwargs):
        # Exercise the real CLI parser/config resolution, without downloading a
        # checkpoint or substituting a model or synthesized speech.
        resolved.update(inspect.currentframe().f_back.f_locals)
        raise ModelLoadReached

    monkeypatch.setattr(utils_infer, "load_model", stop_before_model_load)
    with pytest.raises(ModelLoadReached):
        runpy.run_module("wavtts.infer.infer_cli", run_name="__main__")

    assert resolved[name] == (0.75 if value is None else value)
