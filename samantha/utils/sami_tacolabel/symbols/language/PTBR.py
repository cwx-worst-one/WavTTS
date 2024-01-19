"""
d͡ʒ	consonant
t͡ʃ	consonant
ũ	vowel_nasal
ɐ̃	vowel_nasal
ʊ̃	vowel_nasal
æ	vowel_strong
b	consonant
ɡ	consonant
ɲ	consonant
ʃ	consonant
ʎ	consonant
d	consonant
e	vowel_strong
f	consonant
i	vowel_weak
j	vowel_weak
k	consonant
l	consonant
m	consonant
n	consonant
ŋ	consonant
o	vowel_strong
p	consonant
r	consonant
s	consonant
t	consonant
u	vowel_weak
v	consonant
ɪ	vowel_weak
w	consonant
x	consonant
z	consonant
ʒ	consonant
"""

consonant = [
    "PB0d͡ʒ",
    "PB0t͡ʃ",
    "PB0b",
    "PB0ɡ",
    "PB0ɲ",
    "PB0ʃ",
    "PB0ʎ",
    "PB0d",
    "PB0f",
    "PB0k",
    "PB0l",
    "PB0m",
    "PB0n",
    "PB0ŋ",
    "PB0p",
    "PB0r",
    "PB0s",
    "PB0t",
    "PB0v",
    "PB0w",
    "PB0x",
    "PB0z",
    "PB0ʒ",
]

vowel_nasal = ["PB0ũ", "PB0ɐ̃", "PB0ʊ̃"]
vowel_strong = ["PB0a", "PB0ɔ", "PB0ɛ", "PB0æ", "PB0e", "PB0o"]
vowel_weak = ["PB0i", "PB0j", "PB0u", "PB0ɪ"]
vowel = vowel_nasal + vowel_strong + vowel_weak


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
