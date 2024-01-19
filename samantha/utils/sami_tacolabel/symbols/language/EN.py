consonant = [
    "E0b",
    "E0ch",
    "E0d",
    "E0dh",
    "E0f",
    "E0g",
    "E0h",
    "E0hh",
    "E0jh",
    "E0k",
    "E0l",
    "E0m",
    "E0n",
    "E0p",
    "E0r",
    "E0s",
    "E0sh",
    "E0t",
    "E0th",
    "E0v",
    "E0w",
    "E0y",
    "E0z",
    "E0zh",
]

vowel = [
    "E0aa",
    "E0ae",
    "E0ah",
    "E0ao",
    "E0aw",
    "E0ax",
    "E0ay",
    "E0eh",  # cmu_dict
    "E0er",
    "E0ey",
    "E0ih",
    "E0iy",
    "E0ng",
    "E0ow",
    "E0oy",
    "E0uh",
    "E0uw",  # cmu_dict
    "E0en",
]

phone_set = consonant + vowel

punctuation = {
    "full_stop": ["."],
    "comma": [","],
    "colon": [":"],
    "semicolon": [";"],
    "question_mark": ["?"],
    "exclamation_mark": ["!"],
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

tone_set = ["0", "10", "11", "12"]

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
