class UttError(Exception):
    pass


def parse_utterances(meta: dict, confidence_threshold: float = 0.0) -> list[dict]:
    lyrics_labeled = meta.get("lyrics_labeled", {})

    if lyrics_labeled:
        # This can be done in the data pipeline, but this is a quick fix
        if lyrics_labeled.get("discard", "no") == "yes":
            raise UttError("Discard because of lyrics_labeled.discard=yes")

        utterances = meta.get("lyrics_force_align", {})
        if not utterances:
            return []  # allow for instrumental music

        if len(utterances) == 0:
            raise UttError("Discard because of empty lyrics_force_align")

        temp_utterances = []
        for utt_item in utterances:
            utt_item["attribute"] = {"confidence": 1.0}
            if (
                utt_item.get("singer_tag", "")
                and utt_item.get("phoneme_v86", "")
                and utt_item.get("text", "")
            ):

                singer_tag = utt_item["singer_tag"].replace("：", ":").split(":")[0]
                utt_item["text"] = singer_tag + ":" + utt_item["text"]
            temp_utterances.append(utt_item)
        return temp_utterances
    else:
        lyrics = meta.get("lyrics", {})
        if not lyrics:
            return []  # allow for instrumental music

        # This can be done in the data pipeline, but this is a quick fix
        if not isinstance(lyrics, dict):
            raise UttError("Discard because of lyrics is not a dict")

        if lyrics.get("confidence", 1.0) < confidence_threshold:
            raise UttError(
                "Discard because of lyrics confidence is lower than threshold"
            )

        _result = lyrics.get("result", [])
        _asr_utterances = lyrics.get("utterances")

        if len(_result) != 1 and _asr_utterances is None:
            raise UttError("Discard because of lyrics is empty or invalid results")

        utterances = _result[0].get("utterances") if _result else _asr_utterances
        if not utterances:
            raise UttError("Discard because of No utterances")

        return utterances
