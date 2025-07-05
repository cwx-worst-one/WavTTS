class UttError(Exception):
    pass


def parse_utterances(meta: dict, confidence_threshold: float = 0.0) -> list[dict]:
    lyrics = meta.get("lyrics", {})
    if not lyrics:
        return []  # allow for instrumental music

    # This can be done in the data pipeline, but this is a quick fix
    if not isinstance(lyrics, dict):
        raise UttError("Discard because of lyrics is not a dict")

    if lyrics.get("confidence", 1.0) < confidence_threshold:
        raise UttError("Discard because of lyrics confidence is lower than threshold")

    _result = lyrics.get("result", [])
    _asr_utterances = lyrics.get("utterances")

    if len(_result) != 1 and _asr_utterances is None:
        raise UttError("Discard because of lyrics is empty or invalid results")

    utterances = _result[0].get("utterances") if _result else _asr_utterances
    if not utterances:
        raise UttError("Discard because of No utterances")

    return utterances
