# MIT License

# Copyright (c) 2022 OpenAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import unicodedata

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s
