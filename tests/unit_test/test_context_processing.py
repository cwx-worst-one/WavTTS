''' test context processing. '''

import io
import pickle
from subword_nmt.apply_bpe import BPE as ApplyBPE
from transformers import BertTokenizer as BertTokenizer_huggingface
from dataloader import FalconReader
from core.dataset import get_meta
from core.dataset.preprocess import (
    BPE,
    DialogHistToContext,
    Textchar2Index,
    BertTokenizer,
)
from core.utils.dist_hdfs import dist_hdfs_get
from core.models.pretrained.bert_utils import build_bert_vocab


def test_context_processing():
    '''test function for context processing'''
    # pylint: disable=line-too-long
    meta_file = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/resources/en_meta_data/swbd_meta_vocab2k_fs8k'
    meta_data = get_meta(meta_file)
    reorder_tgt_dict = meta_data['reorder_tgt_dict']
    total_code = meta_data['total.code']
    bpe_fn = ApplyBPE(io.StringIO(total_code))

    # pylint: disable=line-too-long
    reader = FalconReader(
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/datasets/english/fisher_2000h/training_nodev/shard_0',
        10,
    )
    _keys = reader.list_keys()

    # pylint: disable=line-too-long
    bert_vocab_dict = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/houjunfeng/saved_models/dolphin_tools/context_aware_rnnt/bert_init_model/english/vocab.txt'
    local_vocab_file = dist_hdfs_get(bert_vocab_dict, './tmp/', 'vocab_dict.txt')
    bert_tokenizer = BertTokenizer_huggingface(local_vocab_file)

    dialog_history = DialogHistToContext(
        key='dialogue_history',
        out_key='context_text',
        turns=3,
        max_context_len=256,
        mask_prob=0.1,
        mask_token='<pad>',
        perturb_prob=0.1,
        edit_ops_probs='0.6,0.2,0.2',  # prob of Sub, Ins, Del
        vocab=reorder_tgt_dict.symbols,
    )

    bpe0 = BPE(
        key='context_text',
        out_key='context_bpe',
        bpe_fn=bpe_fn,
        reorder_tgt_dict=reorder_tgt_dict,
        use_eos=False,
        skip_list=['<s>', '</s>', '<pad>'],
    )
    bpe1 = BPE(
        key='label',
        out_key='char',
        bpe_fn=bpe_fn,
        reorder_tgt_dict=reorder_tgt_dict,
        use_eos=False,
        skip_list=['<s>', '</s>', '<pad>'],
    )

    bert_char2index = True
    bert_filter_punc = False
    bert_text_lexicon, _ = build_bert_vocab(bert_vocab_dict, bert_char2index, bert_filter_punc)
    tokenizer_0 = Textchar2Index(
        key='context_text',
        out_key='context_tk0',
        vocab_dict=bert_text_lexicon,
        char2index=True,
        unsqueeze=False,
    )
    tokenizer_1 = BertTokenizer(
        key='context_text', out_key='context_tk1', bert_tokenizer=bert_tokenizer
    )
    vals = reader.read_many([0], True)[0]
    for i, val in enumerate(vals):
        item_data = pickle.loads(val)
        print(i, 'uttid ' + item_data['uttid'])
        print('ori label:\n', item_data['label'])
        print('ori context"\n', item_data.get('context_text'))

        item_data = dialog_history(item_data)
        print('dialogue history:\n', item_data['dialogue_history'])
        print('dialogue history to context:\n', item_data['context_text'])

        item_data = bpe0(item_data)
        item_data = bpe1(item_data)
        print('BPE label:\n', item_data['char'])
        print('BPE context:\n', item_data['context_bpe'])

        print('input context:\n', item_data['context_text'])
        item_data = tokenizer_0(item_data)
        print('Textchar2Index context:\n', item_data['context_tk0'])

        print('input context:\n', item_data['context_text'])
        item_data = tokenizer_1(item_data)
        print('huggingface tokenized context:\n', item_data['context_tk1'])
        print('')
