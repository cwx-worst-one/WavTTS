import json

def is_confident_lyrics(utterance, threshold):
    conf, num_utt = 0., 0.
    for utt in utterance:
        if utt['text'].strip():
            conf += float(utt["confidence"])
            num_utt += 1
    if num_utt == 0:
        return False
    conf /= num_utt
    return True if conf > threshold else False


def test_parquet_dataset():
    from samantha.dataio.parquet import ParquetDataset
    from recipes.bigmusic.datasets.svs import group_utterances
    from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id

    dataset = ParquetDataset(data_id=1798)
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    meta = json.loads(sample["meta"])
    utterances = meta["lyrics"]["result"][0]["utterances"]
    print(is_confident_lyrics(utterances, 0.8))
    segments = group_utterances(utterances, 20, 30, False, False)
    phone_tone_ids, phones, tones = convert_labels_to_text_id(utterances[0]["phoneme"].split("\n"))

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip