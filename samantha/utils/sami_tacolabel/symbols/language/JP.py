# coding=utf-8
"""
有效音素集：[
    'a', 'b', 'ch', 'cl', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n',
    'o', 'p', 'r', 's', 'sh', 't', 'ts', 'u', 'v', 'wa', 'we', 'wi', 'wo',
    'ya', 'ye', 'yo', 'yu', 'z', 'I', 'N', 'U', 'A', 'E', 'O'
]
consonant:[
    'b', 'ch', 'cl', 'd', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p',
    'r', 's', 'sh', 't', 'ts', 'v', 'z', 'N'
]
vowel: [
    'a', 'e', 'i', 'o', 'u', 'wa', 'we', 'wi', 'wo', 'ya', 'ye',
    'yo', 'yu', 'I', 'U', 'A', 'E', 'O'
]
"""

consonant = [
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "v",
    "w",
    "y",
    "z",
    "ty",
    "N",
    "tw",
    "dw",
    "kw",
    "gw",
]
vowel = ["a", "e", "i", "o", "u", "I", "U", "A", "E", "O"]
phone_set = consonant + vowel

punctuation = {
    "full_stop": ["。"],
    "comma": ["，"],
    "colon": ["："],
    "semicolon": ["；"],
    "question_mark": ["？"],
    "exclamation_mark": ["！"],
    "at": ["@"],
    "apostrophe": ["'"],
    "hyphen": ["-"],
    "dash": ["--"],
    "ellipsis": ["..."],
    "dollar": ["$"],
    "slash": ["/"],
    "backslash": ["\\"],
    "quotation_mark": ['"'],
    "parallel": ["||", "|"],
    "ampersand": ["&"],
    "swung_dash": ["~"],
    "sharp": ["#"],
    "pipe": ["|"],
    "percent": ["%"],
    "brace": ["<", ">", "(", ")", "[", "]", "{", "}"],
}

filter_set = ["brace", "quotation_mark", "apostrophe"]

tone_set = ["0", "1", "2"]

pause_set = [
    "ampersand",
    "hyphen",
    "dash",
    "sharp",
    "pipe",
    "at",
    "slash",
    "backslash",
    "percent",
    "dollar",
    "colon",
]

separation_set = ["hyphen", "dash", "ellipsis", "swung_dash", "comma"]

silence_set = ["full_stop", "question_mark", "exclamation_mark"]

punc_phone = ["full_stop", "question_mark", "exclamation_mark"]
