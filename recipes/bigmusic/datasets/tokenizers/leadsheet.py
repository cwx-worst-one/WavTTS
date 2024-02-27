import torch
import numpy as np
from recipes.datasets.mcc.sami_tokenizer import all_phones


all_phones = all_phones + ["ext"]

class FloatTokenizer():
    '''
    This class tokenizes any float into a finite list of integer tokens.
    It first scales the input float by 10**decimal_places and converts it to an integer.
    For example, int(23.013 * 10 ** 3) = 23013

    Then, for each digit, it takes the digit position + the digit itself as an index.
    For instance, 23013 = 20000 + 3000 + 10 + 3
    Because 20000 = 10**4 * 2, it can be expressed as 4+2 = 42

    The integer token is then flattened into a list.
    For example, 23013 = {4}{2} + {3}{3} + {2}1 + 3
                 = [42, 33, 21, 3]
    '''
    def __init__(self, integer_places=3, decimal_places=3, index_offset=1):
        self.scale = 10 ** decimal_places
        self.range = integer_places + decimal_places
        self.index_offset = index_offset
        self.float2token, self.token2float = self.build_vocab(index_offset)

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self, index_offset=1):
        '''
        Builds a vocabulary with a given index_offset.
        The index_offset defaults to 1, meaning the vocabulary index starts from 1 to vocab_size+1.
        '''
        vocab_index = index_offset
        token2float = {vocab_index:0}
        float2token = {0:vocab_index}
        vocab_index += 1

        for pos in range(self.range):
            for integer in range(1, 10):
                value = integer * (10**pos)
                token_shifted = vocab_index
                token2float[token_shifted] = value
                float2token[value] = token_shifted
                vocab_index += 1
        return float2token, token2float

    def tokenize(self, inputs):
        if isinstance(inputs, list):
            tokens = []
            for num in inputs:
                segments = self._split_num(num)
                segments = [self.float2token[t] for t in segments]
                tokens.append(segments)
        else:
            num = inputs
            segments = self._split_num(num)
            tokens = [self.float2token[t] for t in segments]
        
        return tokens
    
    def _split_num(self, num):
        n = int(num*self.scale)
        # Initialize empty list for tokens
        tokens = []
        if n == 0:
            return [0]
            
        # Loop over each digit in the string
        while n >0:
            pos = 10 ** (len(str(n))-1)
            mod = n % pos
            tokens.append(n-mod)
            n = mod
        return tokens

class LeadSheetTokenizer():
    '''
    This class expands and tokenizes a lead sheet into a 1D list of tokens. 
    It builds timestamp_tokenizer, note tokenizer and phoneme tokenizer accordingly.

    Input should be a lead sheet, which is a list of tuples (note, [phonemes], start_time, end_time):
    For example,
    leadsheet = [('Rest', ['sil'],           0.0,   0.5), 
                 ('B3',   ['C0d', 'C0eng'],  0.5,   0.975), 
                 ('Db4',  ['C0g', 'C0uang'], 0.975, 1.4)]

    For each note, we expand as:
    note_token + phone_tokens + '[START_OF_TIME]' +  duration_tokens + '[END_OF_NOTE]' + [next_note_token...
    output = [414, 60, 390, 0, 416, 35, 415, 99, 376, 165, 35, 416, 39, 27, 15, 415, 103, 303, 274, 39, 27, 15, 416, 41, 34, 415, 417]
    '''

    def __init__(self, all_phones=all_phones, timestamp_demical=3, 
                       speical_token=['[START_OF_TIME]', '[DURATION]', '[END_OF_TIME]', '[END_OF_NOTE]'],
                       ):
        self.all_phones = all_phones
        self.speical_token = speical_token

        # usually, 0 index is use for padding, save it to avoid conflict
        self.vocab = {"[PAD]":0}
        self.pad_token_id = 0
        
        # build phone tokenizer
        self.phone_to_token = {phone: i+len(self.vocab) for i, phone in enumerate(self.all_phones)}
        self.vocab.update(self.phone_to_token)
        
        # build note tokenizer
        self.note_to_token = {note: i+len(self.vocab) for i, note in enumerate(self._generate_note_vocab())}
        self.token_to_note = {v:k for k, v in self.note_to_token.items()}
        self.vocab.update(self.note_to_token)

        # build float tokenizer for timestamp
        self.timestamp_tokenizer = FloatTokenizer(decimal_places=timestamp_demical, index_offset=len(self.vocab))
        self.vocab.update(self.timestamp_tokenizer.float2token)

        # build special tokenizer
        self.vocab.update({special_token: i+len(self.vocab) for i, special_token in enumerate(speical_token)})
        self.vocab_size = len(self.vocab)
        
    def _generate_note_vocab(self):
        # https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        # Generate a list of notes according to full MIDI note (128 note)
        # With middle C defined as C4 :
        sym = [f"{note}{octave}" for octave in range(-1, 10) for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]
        sym = sym[:-4] + ["Rest"] # extended MIDI range (128 note + rest)
        self.pitchid2sym = {i:s for i, s in enumerate(sym)}
        self.sym2pitchid = {s:i for i, s in enumerate(sym)}
        return sym

    def get_pitch_ovlp(self, query, key):

        pitch_query = []
        for i in query:
            i = i['pitch']
            if isinstance(i, str) and i != "Rest":
                pitch_query.append(self.normalize_sym(i))

        pitch_key = []
        for i in key:
            i = i['pitch']
            if isinstance(i, str) and i != "Rest":
                pitch_key.append(self.normalize_sym(i))
                
        if len(pitch_query) == 0 or len(pitch_key) == 0:
            return 0.0

        query_range = np.max(pitch_query) - np.min(pitch_query)
        ovlp_range = min(np.max(pitch_query), np.max(pitch_key)) - max(np.min(pitch_query), np.min(pitch_key)) 
        ovlp_range = max(ovlp_range, 0)
        ovlp_rate = ovlp_range / query_range

        return ovlp_rate

    def normalize_sym(self, sym, pitch_shift=0):
        if isinstance(sym, int): 
            pitch_id = sym + pitch_shift
            assert(pitch_id <= 127 and pitch_id >=0) # inside midi pitch id [0,127] + rest
            sym = self.pitchid2sym[pitch_id]
        if sym in self.note_to_token.keys():
            return self.note_to_token[sym]
        # sym must in self.sym2pitchid
        if ('#' not in sym) and ('b' not in sym):
            return self.note_to_token[sym]

        if '#' in sym:
            shift_note = '#'
            n_alter = sym.count(shift_note)
        else:  # 'b' in sym
            shift_note = 'b'
            n_alter = -sym.count(shift_note)

        assert abs(n_alter) <= 2  # only allow like C#4/C##4/Cb4/Cbb4, not allow C###4

        sym_base = sym.replace(shift_note * abs(n_alter), '')
        altered_id = self.note_to_token[sym_base] + n_alter + pitch_shift
        assert altered_id in self.token_to_note.keys()
        return altered_id

    def _tokenize_note(self, notes, pitch_shift=0):
        if not isinstance(notes, list):
            notes = [notes]

        return [self.normalize_sym(note, pitch_shift=pitch_shift) for note in notes]


    def _tokenize_phones(self, phone_ids):
        flattened = []
        for sublist in phone_ids:
            if isinstance(sublist, list):
                flattened.extend(sublist)
            else:
                flattened.append(sublist)
        return [self.phone_to_token[phone] for phone in flattened]

    def _tokenize_time(self, start_time, end_time):
        time_tokens = []
        # Convert start time and duration to their token representations.
        duration = end_time - start_time
        start_time, duration, end_time = self.timestamp_tokenizer.tokenize([start_time, duration, end_time])
        if "start" in self.time_format:
            time_tokens += [self.vocab['[START_OF_TIME]']] 
            time_tokens += start_time
        if "duration" in self.time_format:
            time_tokens += [self.vocab['[DURATION]']] 
            time_tokens += duration
        if "end" in self.time_format:
            time_tokens += [self.vocab['[END_OF_TIME]']] 
            time_tokens += end_time
        return time_tokens

    def _tokenize(self, leadsheet: list, pitch_shift=0) -> list:
        # Tokenize the entire lead sheet.
        tokens = []
        phoneme_tokens = []
        note_tokens = []
        for item in leadsheet:
            note = item["pitch"]
            phone_ids = item["phone"]
            start_time = item["start"]
            end_time = item["end"]
            note_token = self._tokenize_note(note, pitch_shift=pitch_shift)
            phoneme_token = self._tokenize_phones(phone_ids)
            time_tokens = self._tokenize_time(start_time, end_time)

            # For each note, we expand as:
            # note_token + phoneme_token + '[START_OF_TIME]' +  duration_tokens 
            note = note_token + phoneme_token + time_tokens + [self.vocab['[END_OF_NOTE]']]
            phoneme_tokens.extend(phoneme_token)
            note_tokens.extend(note_token)
            tokens.extend(note)

        return tokens, phoneme_tokens, note_tokens

    def __call__(self, leadsheet: list, return_dict=False, pitch_shift=0) -> torch.Tensor:
        leadsheet_tokens, phoneme_tokens, note_tokens = self._tokenize(leadsheet, pitch_shift)
        if return_dict:
            return {
                "leadsheet_tokens": torch.tensor(leadsheet_tokens),
                "phoneme_tokens": torch.tensor(phoneme_tokens),
                "note_tokens": torch.tensor(note_tokens),
            }
        else:
            return torch.tensor(leadsheet_tokens)


class LeadSheetTokenizerV2(LeadSheetTokenizer):
    '''
    This class expands and tokenizes a lead sheet into a 1D list of tokens. 
    It builds timestamp_tokenizer, note tokenizer and phoneme tokenizer accordingly.

    Input should be a lead sheet, which is a list of tuples (note, [phonemes], start_time, end_time):
    For example,
    leadsheet = [('Rest', ['sil'],           0.0,   0.5), 
                 ('B3',   ['C0d', 'C0eng'],  0.5,   0.975), 
                 ('Db4',  ['C0g', 'C0uang'], 0.975, 1.4)]

    For each note, we expand as:
    note_token + phone_tokens + '[START_OF_TIME]' +  duration_tokens + '[END_OF_NOTE]' + [next_note_token...
    output = [414, 60, 390, 0, 416, 35, 415, 99, 376, 165, 35, 416, 39, 27, 15, 415, 103, 303, 274, 39, 27, 15, 416, 41, 34, 415, 417]
    '''

    def __init__(self, all_phones=all_phones, timestamp_demical=3, 
                        time_format = "start,duration",
                       speical_token=['[START_OF_TIME]', '[DURATION]', '[END_OF_TIME]', '[END_OF_NOTE]'],
                       ):
        self.all_phones = all_phones
        self.speical_token = speical_token
        self.time_format = time_format

        # usually, 0 index is use for padding, save it to avoid conflict
        self.vocab = {"[PAD]":0, "[EOS]":1}
        self.pad_token_id = 0
        
        # build phone tokenizer
        self.phone_to_token = {phone: i+len(self.vocab) for i, phone in enumerate(self.all_phones)}
        self.vocab.update(self.phone_to_token)
        
        # build note tokenizer
        self.note_to_token = {note: i+len(self.vocab) for i, note in enumerate(self._generate_note_vocab())}
        self.token_to_note = {v:k for k, v in self.note_to_token.items()}
        self.vocab.update(self.note_to_token)

        # build float tokenizer for timestamp
        self.timestamp_tokenizer = FloatTokenizer(decimal_places=timestamp_demical, index_offset=len(self.vocab))
        self.vocab.update(self.timestamp_tokenizer.float2token)

        # build special tokenizer
        self.vocab.update({special_token: i+len(self.vocab) for i, special_token in enumerate(speical_token)})
        self.vocab_size = len(self.vocab)


    def norm_time(self, x, bias, weight):
        return (x + bias) * weight

    def norm_time_log(self, x, weight=1.0):
        return np.log(x + 1) * weight

    def residual_time(self, x, mod=2):
        residual = x % mod
        return residual

    def _tokenize_time(self, start_time, end_time):
        time_float = []
        time_tokens = []
        # Convert start time and duration to their token representations.
        # the layer-norm in transformer will norm the scaling factor to near-identical within the range [−5, 5]
        # we normalize numbers in the text corpus such that they fall within the range [−5, 5] as a preprocessing step before training
        # xVal: https://arxiv.org/pdf/2310.02989.pdf
        duration = end_time - start_time
        if "linear+-4" in self.time_format:
            start_time = self.norm_time(start_time, bias=-20, weight=4/20) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=-20, weight=4/20) # norm [0, 40] to [-4, 4]
        elif "linear+-1" in self.time_format:
            start_time = self.norm_time(start_time, bias=-20, weight=1/20) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=-20, weight=1/20) # norm [0, 40] to [-4, 4]
        elif "linear+4" in self.time_format:
            start_time = self.norm_time(start_time, bias=0, weight=4/40) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=0, weight=4/40) # norm [0, 40] to [-4, 4]
        elif "log1p+4" in self.time_format:
            start_time = self.norm_time_log(start_time, weight=4.0)
            end_time = self.norm_time_log(end_time, weight=4.0)
        elif "log1p+1" in self.time_format:
            start_time = self.norm_time_log(start_time, weight=1.0)
            end_time = self.norm_time_log(end_time, weight=1.0)
        elif "residual2" in self.time_format:
            start_time = self.residual_time(start_time, mod=2)
            end_time = self.residual_time(end_time, mod=2)
        elif "residual1" in self.time_format:
            start_time = self.residual_time(start_time, mod=1)
            end_time = self.residual_time(end_time, mod=1)
        else:
            pass

        if "start" in self.time_format:
            time_float.append(start_time)
            time_tokens.append(self.timestamp_tokenizer.tokenize(2.0)[0]) # use 2.0 to as start time token
        if "duration" in self.time_format:
            time_float.append(duration)
            time_tokens.append(self.timestamp_tokenizer.tokenize(2.0)[0])
        if "end" in self.time_format:
            time_float.append(end_time)
            time_tokens.append(self.timestamp_tokenizer.tokenize(2.0)[0])
        return time_tokens, time_float

    def _tokenize(self, leadsheet: list, pitch_shift=0) -> list:
        # Tokenize the entire lead sheet.
        tokens = []
        phoneme_tokens = []
        note_tokens = []
        token_coff = []
        for item in leadsheet:
            note = item["pitch"]
            phone_ids = item["phone"]
            start_time = item["start"]
            end_time = item["end"]
            note_token = self._tokenize_note(note, pitch_shift=pitch_shift)
            phoneme_token = self._tokenize_phones(phone_ids)
            time_tokens, time_float = self._tokenize_time(start_time, end_time)

            # For each note, we expand as:
            # note_token + phoneme_token + '[START_OF_TIME]' +  duration_tokens 
            note = note_token + phoneme_token + time_tokens + [self.vocab['[END_OF_NOTE]']]
            token_coff_this = len(note_token) * [1] + len(phoneme_token) * [1]  + time_float + [1]
            phoneme_tokens.extend(phoneme_token)
            note_tokens.extend(note_token)
            tokens.extend(note)
            token_coff.extend(token_coff_this)
        return tokens, phoneme_tokens, note_tokens, token_coff

    def __call__(self, leadsheet: list, return_dict=True, pitch_shift=0) -> torch.Tensor:
        if return_dict == False:
            raise NotImplementedError
        leadsheet_tokens, phoneme_tokens, note_tokens, token_coff = self._tokenize(leadsheet, pitch_shift)
        return {
            "leadsheet_tokens": torch.tensor(leadsheet_tokens),
            "phoneme_tokens": torch.tensor(phoneme_tokens),
            "note_tokens": torch.tensor(note_tokens),
            "token_coff": torch.tensor(token_coff),
        }



class LeadSheetTokenizerV3(LeadSheetTokenizer):
    '''
    This class expands and tokenizes a lead sheet into a 1D list of tokens. 
    It builds timestamp_tokenizer, note tokenizer and phoneme tokenizer accordingly.

    Input should be a lead sheet, which is a list of tuples (note, [phonemes], start_time, end_time):
    For example,
    leadsheet = [('Rest', ['sil'],           0.0,   0.5), 
                 ('B3',   ['C0d', 'C0eng'],  0.5,   0.975), 
                 ('Db4',  ['C0g', 'C0uang'], 0.975, 1.4)]

    Output:
    
    leadsheet_tokens: (Note that we use same token [NUM] for start_time_token and duration_token)
        list of int: [note_token + phone_tokens + start_time_token +  duration_token + next_note_token...]
    token_coff: identicator for each token, same shape with leadsheet_tokens, if token is a float number, take float number itself
        If it's a token, will set to 1.0
        list of float: 
        [1.0, 1.0, 1.0, 0.0(start_time), ]...
    '''

    def __init__(self, all_phones=all_phones, timestamp_demical=3, 
                        time_format = "start,duration",
                       speical_token=['[START_OF_TIME]', '[DURATION]', '[END_OF_TIME]', '[END_OF_NOTE]', '[NUM]'],
                       ):
        self.all_phones = all_phones
        self.speical_token = speical_token
        self.time_format = time_format

        # usually, 0 index is use for padding, save it to avoid conflict
        self.vocab = {"[PAD]":0, "[EOS]":1}
        self.pad_token_id = 0
        
        # build phone tokenizer
        self.phone_to_token = {phone: i+len(self.vocab) for i, phone in enumerate(self.all_phones)}
        self.vocab.update(self.phone_to_token)
        
        # build note tokenizer
        self.note_to_token = {note: i+len(self.vocab) for i, note in enumerate(self._generate_note_vocab())}
        self.token_to_note = {v:k for k, v in self.note_to_token.items()}
        self.vocab.update(self.note_to_token)

        # build special tokenizer
        self.vocab.update({special_token: i+len(self.vocab) for i, special_token in enumerate(speical_token)})
        self.vocab_size = len(self.vocab)


    def norm_time(self, x, bias, weight):
        return (x + bias) * weight

    def norm_time_log(self, x, weight=1.0):
        return np.log(x + 1) * weight

    def residual_time(self, x, mod=2):
        residual = x % mod
        return residual

    def _tokenize_time(self, start_time, end_time):
        time_float = []
        time_tokens = []
        # Convert start time and duration to their token representations.
        # Tried different normalizations before training, but no normalization at all perform best
        # xVal: https://arxiv.org/pdf/2310.02989.pdf
        duration = end_time - start_time
        if "linear+-4" in self.time_format:
            start_time = self.norm_time(start_time, bias=-20, weight=4/20) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=-20, weight=4/20) # norm [0, 40] to [-4, 4]
        elif "linear+-1" in self.time_format:
            start_time = self.norm_time(start_time, bias=-20, weight=1/20) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=-20, weight=1/20) # norm [0, 40] to [-4, 4]
        elif "linear+4" in self.time_format:
            start_time = self.norm_time(start_time, bias=0, weight=4/40) # norm [0, 40] to [-4, 4]
            end_time = self.norm_time(end_time, bias=0, weight=4/40) # norm [0, 40] to [-4, 4]
        elif "log1p+4" in self.time_format:
            start_time = self.norm_time_log(start_time, weight=4.0)
            end_time = self.norm_time_log(end_time, weight=4.0)
        elif "log1p+1" in self.time_format:
            start_time = self.norm_time_log(start_time, weight=1.0)
            end_time = self.norm_time_log(end_time, weight=1.0)
        elif "residual2" in self.time_format:
            start_time = self.residual_time(start_time, mod=2)
            end_time = self.residual_time(end_time, mod=2)
        elif "residual1" in self.time_format:
            start_time = self.residual_time(start_time, mod=1)
            end_time = self.residual_time(end_time, mod=1)
        else:
            # default setting
            pass

        if "start" in self.time_format:
            time_float.append(start_time)
            time_tokens.append(self.vocab['[NUM]'])
        if "duration" in self.time_format:
            time_float.append(duration)
            time_tokens.append(self.vocab['[NUM]'])
        if "end" in self.time_format:
            time_float.append(end_time)
            time_tokens.append(self.vocab['[NUM]'])
        return time_tokens, time_float

    def _tokenize(self, leadsheet: list, pitch_shift=0) -> list:
        # Tokenize the entire lead sheet.
        tokens = []
        phoneme_tokens = []
        note_tokens = []
        token_coff = []
        for item in leadsheet:
            note = item["pitch"]
            phone_ids = item["phone"]
            start_time = item["start"]
            end_time = item["end"]
            note_token = self._tokenize_note(note, pitch_shift=pitch_shift)
            phoneme_token = self._tokenize_phones(phone_ids)
            time_tokens, time_float = self._tokenize_time(start_time, end_time)

            # For each note, we expand as:
            # note_token + phoneme_token + '[START_OF_TIME]' +  duration_tokens 
            note = note_token + phoneme_token + time_tokens + [self.vocab['[END_OF_NOTE]']]
            token_coff_this = len(note_token) * [1] + len(phoneme_token) * [1]  + time_float + [1]
            phoneme_tokens.extend(phoneme_token)
            note_tokens.extend(note_token)
            tokens.extend(note)
            token_coff.extend(token_coff_this)
        return tokens, phoneme_tokens, note_tokens, token_coff

    def __call__(self, leadsheet: list, return_dict=True, pitch_shift=0) -> torch.Tensor:
        if return_dict == False:
            raise NotImplementedError
        leadsheet_tokens, phoneme_tokens, note_tokens, token_coff = self._tokenize(leadsheet, pitch_shift)
        return {
            "leadsheet_tokens": torch.tensor(leadsheet_tokens),
            "phoneme_tokens": torch.tensor(phoneme_tokens),
            "note_tokens": torch.tensor(note_tokens),
            "token_coff": torch.tensor(token_coff),
        }


def test():
    leadsheet = [('Rest', ['sil'],           0.0,   0.5), 
                 ('Bb3',   ['C0d', 'C0eng'],  0.5,   0.975), 
                 ('D#4',  ['C0g', 'C0uang'], 0.975, 1.4), 
                 ('D#4',  ['C0g', 'C0uang'], 1.4, 21.4), 
                 ('D#4',  ['C0g', 'C0uang'], 21.4, 33.4),]


    leadsheet_format = []
    for note, phone_ids, start_time, end_time in leadsheet:
        leadsheet_format.append({
                                "start":start_time, 
                                "end":end_time, 
                                "phone":phone_ids, 
                                "pitch":note, 
                                })

    # leadsheet_format = [{'start': 28.84, 'end': 29.22, 'pitch': 67, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 29.28, 'end': 30.16, 'pitch': 65, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 30.240000000000002, 'end': 30.92, 'pitch': 60, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 31.04, 'end': 31.46, 'pitch': 58, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 32.18, 'end': 32.42, 'pitch': 58, 'phone': ['sil', 'E0w', 'E0iy']}, {'start': 32.46, 'end': 32.82, 'pitch': 58, 'phone': ['sil', 'E0g', 'E0aa']}, {'start': 32.9, 'end': 33.26, 'pitch': 67, 'phone': ['sil', 'E0t', 'E0uw']}, {'start': 33.34, 'end': 33.82, 'pitch': 65, 'phone': ['sil', 'E0l', 'E0ah']}, {'start': 33.82, 'end': 34.1, 'pitch': 63, 'phone': ['sil', 'E0t', 'E0uw']}, {'start': 34.1, 'end': 34.7, 'pitch': 62, 'phone': ['sil', 'E0l', 'E0ih']}, {'start': 34.84, 'end': 35.12, 'pitch': 53, 'phone': ['E0g', 'E0eh']}, {'start': 35.12, 'end': 35.4, 'pitch': 55, 'phone': ['E0g', 'E0eh']}, {'start': 35.4, 'end': 35.78, 'pitch': 58, 'phone': ['E0dh', 'E0er']}, {'start': 36.52, 'end': 36.94, 'pitch': 67, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 36.94, 'end': 37.88, 'pitch': 65, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 37.94, 'end': 38.660000000000004, 'pitch': 60, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 38.76, 'end': 39.2, 'pitch': 58, 'phone': ['sil', 'E0ch', 'E0uw']}, {'start': 40.84, 'end': 41.160000000000004, 'pitch': 62, 'phone': ['sil', 'E0y', 'E0uw']}, {'start': 41.160000000000004, 'end': 41.52, 'pitch': 65, 'phone': ['sil', 'E0ae']}, {'start': 41.62, 'end': 42.0, 'pitch': 65, 'phone': ['sil', 'E0m', 'E0iy']}, {'start': 42.08, 'end': 42.56, 'pitch': 62, 'phone': ['sil', 'E0t', 'E0ah']}, {'start': 42.56, 'end': 43.04, 'pitch': 67, 'phone': ['E0dh', 'E0er']}, {'start': 45.96, 'end': 46.2, 'pitch': 70, 'phone': ['sil', 'E0w', 'E0ay']}, {'start': 46.32, 'end': 46.52, 'pitch': 70, 'phone': ['sil', 'E0dh', 'E0ah']}, {'start': 46.64, 'end': 46.800000000000004, 'pitch': 70, 'phone': ['sil', 'E0p', 'E0iy']}, {'start': 46.800000000000004, 'end': 46.88, 'pitch': 67, 'phone': ['E0p', 'E0ah']}, {'start': 47.0, 'end': 47.2, 'pitch': 67, 'phone': ['sil', 'E0dh', 'E0ah']}, {'start': 47.32, 'end': 47.54, 'pitch': 67, 'phone': ['sil', 'E0dh', 'E0ah']}, {'start': 47.62, 'end': 47.86, 'pitch': 65, 'phone': ['sil', 'E0p', 'E0iy']}, {'start': 47.94, 'end': 48.120000000000005, 'pitch': 65, 'phone': ['sil', 'E0f', 'E0ay']}, {'start': 48.22, 'end': 48.5, 'pitch': 65, 'phone': ['sil', 'E0f', 'E0ay']}, {'start': 48.5, 'end': 48.78, 'pitch': 62, 'phone': ['sil', 'E0ih']}, {'start': 48.78, 'end': 49.120000000000005, 'pitch': 65, 'phone': ['sil', 'E0s', 'E0ow']}, {'start': 49.22, 'end': 49.76, 'pitch': 67, 'phone': ['sil', 'E0iy']}, {'start': 49.76, 'end': 50.08, 'pitch': 65, 'phone': ['sil', 'E0t', 'E0uw']}, {'start': 50.08, 'end': 50.36, 'pitch': 62, 'phone': ['sil', 'E0t', 'E0uw']}, {'start': 50.36, 'end': 50.660000000000004, 'pitch': 60, 'phone': ['sil', 'E0k', 'E0r', 'E0ih']}, {'start': 50.660000000000004, 'end': 50.92, 'pitch': 58, 'phone': ['E0t', 'E0ah']}, {'start': 50.92, 'end': 51.6, 'pitch': 62, 'phone': ['E0s', 'E0ay']}, {'start': 53.620000000000005, 'end': 53.86, 'pitch': 60, 'phone': ['sil', 'E0t', 'E0uw']}, {'start': 53.92, 'end': 54.32, 'pitch': 60, 'phone': ['sil', 'E0p', 'E0oy']}, {'start': 54.32, 'end': 54.64, 'pitch': 58, 'phone': ['sil', 'E0dh', 'E0ae']}, {'start': 54.64, 'end': 54.94, 'pitch': 60, 'phone': ['sil', 'E0k', 'E0r', 'E0uh']}, {'start': 54.94, 'end': 55.2, 'pitch': 58, 'phone': ['sil', 'E0k', 'E0r', 'E0uh']}, {'start': 55.26, 'end': 55.54, 'pitch': 55, 'phone': ['E0ng']}, {'start': 55.58, 'end': 55.94, 'pitch': 55, 'phone': ['E0g', 'E0er']}, {'start': 55.94, 'end': 56.28, 'pitch': 58, 'phone': ['sil', 'E0l', 'E0ay']}, {'start': 56.32, 'end': 56.94, 'pitch': 58, 'phone': ['sil', 'E0s', 'E0ah']}, {'start': 56.94, 'end': 57.36, 'pitch': 62, 'phone': ['sil', 'E0b', 'E0ae']}, {'start': 57.36, 'end': 57.72, 'pitch': 60, 'phone': ['E0k', 'E0w', 'E0er']}, {'start': 57.76, 'end': 58.1, 'pitch': 60, 'phone': ['E0k', 'E0w', 'E0er']}, {'start': 58.1, 'end': 58.44, 'pitch': 55, 'phone': ['sil', 'E0ae']}, {'start': 58.44, 'end': 59.56, 'pitch': 58, 'phone': ['E0l', 'E0ah']}]
    tokenizer = LeadSheetTokenizerV2(time_format="start,duration_residual1")
    tokens = tokenizer(leadsheet_format)
    print(tokenizer.vocab)

    # tokens = [332,   1, 123, 162,  52,  48,  38,  24, 415,  33,  27,  19, 416, 330,
    #        1, 123, 162,  52,  49,  32,  28, 415,  38,  27,  19, 416, 325,   1,
    #      123, 162,  53,  32,  24, 415,  36,  27,  19, 416, 323,   1, 123, 162,
    #       53,  41,  24, 415,  34,  22, 416, 323,   1, 142, 157,  53,  42,  31,
    #       28, 415,  32,  24, 416, 323,   1, 127, 146,  53,  42,  34,  26, 415,
    #       33,  25,  19, 416, 332,   1, 139, 162,  53,  42,  39, 415,  33,  25,
    #       19, 416, 330,   1, 132, 148,  53,  43,  33,  24, 415,  34,  27,  19,
    #      416, 328,   1, 139, 162,  53,  43,  38,  22, 415,  32,  28, 416, 327,
    #        1, 132, 156,  53,  44,  31, 415,  36, 416, 318, 127, 153,  53,  44,
    #       38,  24, 415,  32,  27,  19, 416, 320, 127, 153,  53,  45,  31,  22,
    #      415,  32,  28, 416, 323, 125, 154,  53,  45,  34, 415,  33,  28, 416,
    #      332,   1, 123, 162,  53,  46,  35,  22, 415,  34,  21,  19, 416, 330,
    #        1, 123, 162,  53,  46,  39,  24, 415,  39,  24, 416, 325,   1, 123,
    #      162,  53,  47,  39,  24, 415,  37,  22, 416, 323,   1, 123, 162,  53,
    #       48,  37,  26, 415,  34,  24, 416, 327,   1, 143, 162,  54,  38,  24,]
    
    print(tokens)
    data = []
    for d in tokens:
        for k,v in tokenizer.vocab.items():
            if v == d:
                data.append(k)
                break
    print(data)

if __name__ == "__main__":
    test()
