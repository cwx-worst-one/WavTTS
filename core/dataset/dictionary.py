""" Dictionary. """
import os
from collections import Counter
from multiprocessing import Pool
import numpy as np
from core.dataset.dict_utils import tokenize_line, safe_readline, process_bpe_symbol


# pylint: disable=missing-function-docstring


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', bos='<s>', extra_special_symbols=None):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, array, bpe_symbol=None, escape_unk=False):
        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            return self[i]

        banned_item = [self.eos(), self.bos(), self.pad()]
        sent = ' '.join(token_string(i) for i in array if i not in banned_item)
        return process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        idx = len(self.symbols)
        self.indices[word] = idx
        self.symbols.append(word)
        self.count.append(n)
        return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(dict(sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.index('<blank>')

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f, ignore_utf_errors)
        return d

    def add_from_file(self, f, ignore_utf_errors=False):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        self.add_from_file(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError as ue:
                raise Exception(
                    "Incorrect encoding detected in {}, please rebuild the dataset".format(f)
                ) from ue
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1 :])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                self.save(fd)
                return
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    # pylint: disable=no-self-use
    def _get_meta(self):
        return [], []

    def _load_meta(self, _lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f, zip(ex_keys + self.symbols[self.nspecial :], ex_vals + self.count[self.nspecial :])
        )

    def dummy_sentence(self, length):
        # t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t = np.zeros((length,), dtype='int32')
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words) + 1 if append_eos else len(words)
        # ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        ids = np.zeros((nwords,), dtype='int32')

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    # pylint: disable=invalid-name
    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dct, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dct.add_symbol(w, c)

        if num_workers > 1:
            # pylint:disable=consider-using-with
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dct.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(filename, tokenize, dct.eos_word)
            )


class ScpDictionary(Dictionary):
    """ScpDictionary."""

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            words = (line.strip().split(' '))[1:]
            line = ''.join(words)
            while line:
                # for word in tokenize(line):
                for word in line:
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
                words = (line.strip().split(' '))[1:]
                line = ''.join(words)
                print("process line %s" % line)
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dct, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dct.add_symbol(w, c)

        if num_workers > 1:
            # pylint:disable=consider-using-with
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        ScpDictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dct.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                ScpDictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dct.eos_word
                )
            )

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = tokenize_line(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        if append_eos and (len(words) == 0 or words[-1] != '</s>'):
            nwords += 1
        # ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        ids = np.zeros((nwords,), dtype='int32')

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos and (len(words) == 0 or words[-1] != '</s>'):
            ids[-1] = self.eos_index
        return ids

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        kv_iterator = zip(
            ex_keys + self.symbols[self.nspecial :], ex_vals + self.count[self.nspecial :]
        )
        with open(f, 'w', encoding='utf-8') as f_handler:
            for k, v in kv_iterator:
                f_handler.write('{} {}\n'.format(k, v))

    @classmethod
    def load(cls, f, f_non_lang_syms=None, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```

        Identifies the space symbol if it exists, by obtaining its index
        (space_index=-1 if no space symbol)

        Loads non_lang_syms from another text file, if it exists, with one
        symbol per line
        """
        d = super().load(f, ignore_utf_errors)
        # d.space_index = d.indices.get(d.space_word, -1)

        d.space_index = d.indices.get(d.pad_index, -1)

        if f_non_lang_syms is not None:
            assert isinstance(f_non_lang_syms, str)
            try:
                with open(
                    f_non_lang_syms,
                    'r',
                    encoding='utf-8',
                    errors='ignore' if ignore_utf_errors else None,
                ) as fd:
                    non_lang_syms = [x.rstrip() for x in fd.readlines()]
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError as ue:
                raise Exception(
                    "Incorrect encoding detected in {}, please rebuild the dataset".format(f)
                ) from ue

            for sym in non_lang_syms:
                assert d.index(sym) != d.unk(), '{} in {} is not in the dictionary'.format(
                    sym, f_non_lang_syms
                )
            d.non_lang_syms = non_lang_syms

        return d

    def tokens_to_sentence(
        self, line, line_tokenizer=tokenize_line, use_unk_sym=True, bpe_symbol=None
    ):
        if bpe_symbol is not None:
            return process_bpe_symbol(line, bpe_symbol)
        # use_unk_sym=False when we want to restore original transcripts from
        # token sequences, e.g., obtain reference to compute WER
        tokens = line_tokenizer(line)
        sent = ""
        # pylint: disable=comparison-with-callable
        self.space_word = self.unk
        for token in tokens:
            if token == self.space_word:
                sent += " "
            elif use_unk_sym and self.index(token) == self.unk_index:
                sent += self.unk_word
            elif token not in (self.pad_word, self.eos_word):
                sent += token
        return sent.strip()
