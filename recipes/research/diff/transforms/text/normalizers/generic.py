from string import punctuation


def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    return text.translate(str.maketrans("", "", nlp_punctuation)).strip()
