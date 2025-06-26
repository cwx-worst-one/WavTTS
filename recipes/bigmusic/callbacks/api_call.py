import json
import msgpack
import requests
from ray.serve import get_serve_http_client
import os
from recipes.bigmusic.utils.upload import retry

@retry(times=3)
def api_call(audio_path, model=None, verbose=True):

    # check inpit
    audio_path = str(audio_path)
    # assert model in ["bigmir-deepchorus2", "bigmir-bigmusic-tag", "bigmir-vocal2midi"]

    PSM = os.environ.get("PSM")

    if PSM is None:
        PSM = "inf.ray.serve_bigspeech_multitrial.service.lf"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Begin to call {model} model...")

    client = get_serve_http_client(psm=PSM, router=model)
    batch = [
        {
            "uttid": "fec0b386-0609-4396-b409-7e3e125d4dac",
            "meta": "",
            "audio": audio_bytes,
        }
    ]
    url = client.get_one_request_url(True)
    resp = requests.post(url, data=msgpack.packb(batch))
    assert resp is not None

    if resp.status_code != 200:
        print(f"Failed to call mir model: {resp.text}")
    else:
        if verbose:
            if len(str(resp.json())) > 500:
                print(str(resp.json())[:500] + "...")
            else:
                print(resp.json())
    
    return resp.json()

    
if __name__ == "__main__":
    api_call("/mnt/bn/music-llm-nas-lq/zoupei/infer_results/250429_v5_rltice/test_upload/20250429-0053034979/cn_lyrics2song_AIGC_inhouse_v2m/cnfli121.generated.wav", model="bigmir-deepchorus2")
