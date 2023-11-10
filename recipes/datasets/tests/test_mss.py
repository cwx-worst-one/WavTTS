import random
from turtle import xcor
import pytest
import torchaudio
from tqdm import tqdm
import os

from samantha.dataio.webdataset.extension import IndexedWebDataset
from recipes.datasets.mcc.mix import MSSTransforms, MSSTransformsV2, MSSDataset, MSSDataModule
from transformers import BertTokenizer

TEST_HDFS_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-blues_url2index.txt"

'''
test_mss.py
11/9/2023 @hanoihantrakul
This test file contains two types of functions. Those beginning with `test` and `run` 
`test_xxx`: is a pytest function for testing the method in an isolated manner.
`run_xxx`: is a main function simulating how the method would be called in a real training loop. 
'''

def _assert_transform_status(transform, expected_count):
    assert transform.count == expected_count 
    assert transform.skipped == 0
    assert len(transform.messages.keys()) == 0

def _assert_dict_contents(results_dict):
    assert results_dict['audio'].size(-1) > 0
    assert results_dict['inst_audio'].size(-1) > 0
    assert results_dict['vocal_audio'].size(-1) > 0
    assert results_dict['tag'] == 'mss'
    assert len(results_dict['utterance_text']) > 0

def test_mss_transform_without_tokenizer():
    transform = MSSTransforms()
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    dataset = iter(dataset)
    x = next(dataset) 

    result_generator = transform(x) # transform(x) returns a generator
    for result in result_generator:
        _assert_dict_contents(result)
    
    # For this test dataset, I know the first 6 utterances will pass by inspection.
    expected_count = 6
    _assert_transform_status(transform, expected_count)
    
def test_mss_transform_with_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    transform = MSSTransforms(tokenizer=tokenizer)
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    dataset = iter(dataset)
    x = next(dataset) 

    result_generator = transform(x) # transform(x) returns a generator
    for result in result_generator:
        assert result['token'].size(-1) > 0

    # For this test dataset, I know the first 6 utterances will pass by inspection.
    expected_count = 6
    _assert_transform_status(transform, expected_count)
    
def test_mss_transform_v2_without_tokenizer():
    transform = MSSTransformsV2()
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    dataset = iter(dataset)
    x = next(dataset) 

    result_generator = transform(x) # transform(x) returns a generator
    for result in result_generator:
        _assert_dict_contents(result)
    
    # For this test dataset, I know the first 6 utterances will pass by inspection.
    expected_count = 6
    _assert_transform_status(transform, expected_count)

def test_mss_transform_v2_with_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    transform = MSSTransformsV2(tokenizer=tokenizer)
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    dataset = iter(dataset)
    x = next(dataset) 

    result_generator = transform(x) # transform(x) returns a generator
    for result in result_generator:
        assert result['token'].size(-1) > 0

    # For this test dataset, I know the first 6 utterances will pass by inspection.
    expected_count = 6
    _assert_transform_status(transform, expected_count)

def test_mss_dataset():
    '''Load dataset, save audio and print contents of outputs. Manually verify audio contents.'''
    num_iterations = 5
    
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    dataset = MSSDataset(tokenizer=tokenizer) 
    sample_rate = dataset.data_sample_rate

    output_dir = './test_mss_dataset'
    os.makedirs(output_dir, exist_ok=True)

    dataset_iter = iter(dataset)
    for i in range(num_iterations):
        batch = next(dataset_iter)
        for audio_key in ['audio', 'vocal_audio', 'inst_audio']:
            audio = batch[audio_key] 
            assert audio.size(-1) > 0
            filename = os.path.join(output_dir, f"{dataset.name}-item-{i}-{audio_key}.wav")
            torchaudio.save(filename, audio, sample_rate)
            
        text = batch.get("utterance_text", None)
        if text is not None:
            assert len(text) > 0
            print(text)
        
        token = batch.get("token", None)
        if token is not None:
            assert len(token) > 0
            print(token)

def run_mss_transform_without_tokenizer():
    '''Run this function for a few minutes to check it works for long datasets. Then manually cancel ctrl+c.'''
    transform = MSSTransforms(log_interval=20) # increase logging frequency for easier manual inspection
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    for x in dataset:
        for result in transform(x): # transform(x) returns a generator that can be iterated over
            pass

def run_mss_transform_with_tokenizer():
    '''Run this function for a few minutes to check it works for long datasets. Then manually cancel ctrl+c.'''
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    transform = MSSTransforms(tokenizer=tokenizer, log_interval=20) # increase logging frequency for easier manual inspection
    dataset = IndexedWebDataset(TEST_HDFS_PATH).decode()
    
    for x in dataset:
        for result in transform(x): # transform(x) returns a generator that can be iterated over
            pass      

def run_mss_datamodule():
    '''
    This is similar to test_mss_dataset() but ensures MSSDataModule() behaves correctly.
    (e.g. checks behavior of MSSDataModule() bucketizer, multi-worker and collate_fn implementation)
    Run for several minutes and inspect printouts.
    '''
    num_iterations = 2
    sample_rate = 24000

    output_dir = './test_mss_datamodule'
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    pl_datamodule = MSSDataModule(
        sample_rate=sample_rate,
        batch_size=sample_rate * 10 * 30,
        shuffle_buffer_size=10,
        num_workers=2,
        tokenizer=tokenizer)
    train_loader = pl_datamodule.train_dataloader()
    train_loader = iter(train_loader)

    for i in range(num_iterations):
        batch = next(train_loader)
        assert set(batch.keys()) == set(['audio','vocal_audio','inst_audio','token'])

        for audio_key in ['audio', 'vocal_audio', 'inst_audio']:
            audios = batch[audio_key] 
            assert audios.size(-1) > 0 # e.g. audios.size = [12, 574440]
            num_audios = audios.shape[0]
            for j in range(num_audios):
                single_audio = audios[j]
                filename = os.path.join(output_dir, f"MSSDataModule-batch-{i}-item-{j}-{audio_key}.wav")
                torchaudio.save(filename, single_audio.view(1, -1), sample_rate, channels_first=True)
        
        token = batch.get("token", None)
        if token is not None:
            assert token.size(-1) > 0
        
if __name__ == "__main__":
    #run_mss_transform_without_tokenizer()
    #run_mss_transform_with_tokenizer()
    run_mss_datamodule()
    #pass
