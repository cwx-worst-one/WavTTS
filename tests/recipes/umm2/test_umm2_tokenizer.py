import os
from recipes.umm2.requires.model_initializer import init_model
import torch
import torch.nn.functional as F
import torchaudio
import math
import pytest

# TEST_CASE_1 = "hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/misc/umm2_test_cases/aigc_test1.wav"
# TEST_CASE_2 = "hdfs://harunawl/home/byte_data_seed_wl/speech/user/rui.xia/misc/umm2_test_cases/inp071.generated.wav"
# CI failed to get apply permission url of dc wlby
TEST_CASE_1 = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/unittest/aigc_test1.wav"
TEST_CASE_2 = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/unittest/inp071.generated.wav"
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def estimate_token_num(audio_length, sample_rate=24000, token_rate=25):
    return math.ceil(int(audio_length / sample_rate) * token_rate)



def load_example_audio(audio_path, device=None):

    audio_file_name = os.path.basename(audio_path)
    os.system(f"hdfs dfs -get {audio_path} .")

    print(f"audio_path:{audio_file_name}")

    audio, sr = torchaudio.load(audio_file_name)
    
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    audio = audio[0].unsqueeze(0).unsqueeze(0).to(device)  # B, T
    audio_length = torch.LongTensor([audio.shape[-1]]).to(device)
    print("audio duration", audio.shape[-1] / 24000)
    
    return audio, audio_length


def run_single_batch(inference_mode, ckpt, module, requires, test_audio_path, max_len):
    
    audio, audio_length = load_example_audio(test_audio_path)
    model = init_model(ckpt, module, device=device)

    if audio_length < max_len:
        audio = F.pad(audio, (0, max_len - audio_length), "constant", 0)
    elif audio_length > max_len:
        audio = audio[..., :max_len]
        audio_length = max_len

    # Use the args to decide which function to call
    if inference_mode == 'samplewise':
        print("--- Running in sample-wise mode ---")
        inference_function = model.wav2requires_samplewise
    else: # 'batchwise'
        print("--- Running in batch-wise mode ---")
        # Assuming the original function is named wav2requires
        inference_function = model.wav2requires 


    #chunk wise inference
    chunk_output_dict = inference_function(
        audio, audio_length, requires, slice_method="even", chunk_size=45
    )

    for k in chunk_output_dict.keys():
        print(k)
        print(chunk_output_dict[k].shape if isinstance(chunk_output_dict[k], torch.Tensor) and chunk_output_dict[k].ndim > 1 else chunk_output_dict[k])

    #full wise inference
    full_output_dict = inference_function(
        audio, audio_length, requires=requires, slice_method="full"
    )

    chunk_tokens = chunk_output_dict["token"]
    full_tokens = full_output_dict["token"]


    return chunk_tokens, full_tokens


def run_multiple_batch(inference_mode, token_rate, ckpt, module, requires):

    token_rate = token_rate

    audio_test_1, audio_length_test_1 = load_example_audio(TEST_CASE_1)
    audio_test_2, audio_length_test_2 = load_example_audio(TEST_CASE_2)
    audio_test_1_est_token_num = estimate_token_num(audio_length_test_1, token_rate=token_rate)
    audio_test_2_est_token_num = estimate_token_num(audio_length_test_2, token_rate=token_rate)

    print("\n--- Before Batching ---")
    print(f"Audio 1 Shape: {audio_test_1.shape}, Length: {audio_length_test_1.item()}")
    print(f"Audio 2 Shape: {audio_test_2.shape}, Length: {audio_length_test_2.item()}")
    print(f"Audio 1 Est Token Num: {audio_test_1_est_token_num}")
    print(f"Audio 2 Est Token Num: {audio_test_2_est_token_num}")

    # 2. Get the max length for padding
    max_len = max(audio_length_test_1.item(), audio_length_test_2.item())
    print(f"\nMax length for padding: {max_len}")

    # 3. Pad the shorter audio tensor
    # The pad format is (padding_left, padding_right) for the last dimension
    pad1 = max_len - audio_length_test_1.item()
    pad2 = max_len - audio_length_test_2.item()

    padded_audio_1 = F.pad(audio_test_1, (0, pad1), "constant", 0)
    padded_audio_2 = F.pad(audio_test_2, (0, pad2), "constant", 0)

    # 4. Combine the padded tensors into a batch
    batch_audio = torch.cat((padded_audio_1, padded_audio_2), dim=0)

    # 5. Combine the length tensors
    batch_lengths = torch.cat((audio_length_test_1, audio_length_test_2), dim=0)


    model = init_model(ckpt, module, device=device)

    # Use the args to decide which function to call
    if inference_mode == 'samplewise':
        print("--- Running in sample-wise mode ---")
        inference_function = model.wav2requires_samplewise
    else: # 'batchwise'
        print("--- Running in batch-wise mode ---")
        # Assuming the original function is named wav2requires
        inference_function = model.wav2requires 

    

    #chunk wise inference
    chunk_output_dict = inference_function(
        batch_audio, batch_lengths, requires, slice_method="even", chunk_size=45
    )

    for k in chunk_output_dict.keys():
        print(k)
        print(chunk_output_dict[k].shape if isinstance(chunk_output_dict[k], torch.Tensor) and chunk_output_dict[k].ndim > 1 else chunk_output_dict[k])

    #full wise inference
    full_output_dict = inference_function(
        batch_audio, batch_lengths, requires=requires, slice_method="full"
    )

    chunk_tokens = chunk_output_dict["token"]
    full_tokens = full_output_dict["token"]


    return chunk_tokens, full_tokens, max_len


@pytest.mark.parametrize("inference_mode", ["samplewise", "batchwise"])
def test_inference_consistency(
                inference_mode,
                ckpt="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/si.luo/umm2/umm_stage3_rmpad_v2/2025070400_UMM-stage3_64GPU_16mins_fused_no_consistencyloss_model/checkpoints/step=0080000.ckpt", 
                module="recipes.umm2.modules.stages.stage2.Stage2", 
                requires=['token', 'loss'], 
                out_full="test_full.txt", 
                out_chunk="test_chunk.txt", 
                token_rate=25):
    
    multiple_data_chunk_token, multiple_data_full_token, max_len = run_multiple_batch(inference_mode, token_rate, ckpt, module, requires)
    test_case_1_single_data_chunk_token, test_case_1_single_data_full_token = run_single_batch(inference_mode, ckpt, module, requires, TEST_CASE_1, max_len)
    test_case_2_single_data_chunk_token, test_case_2_single_data_full_token = run_single_batch(inference_mode, ckpt, module, requires, TEST_CASE_2, max_len)

    #Test case 1 is long audio
    if inference_mode == 'samplewise':
        assert torch.equal(test_case_1_single_data_chunk_token.squeeze(), multiple_data_chunk_token[0])
        assert torch.equal(test_case_1_single_data_full_token.squeeze(), multiple_data_full_token[0])
        print(f"Test case 1 passed! chunk mode and full mode are equal in {inference_mode} mode")
    else:
        diff_full_cnt = torch.nonzero(multiple_data_full_token[0] - test_case_1_single_data_full_token.squeeze()).numel()
        diff_chunk_cnt = torch.nonzero(multiple_data_chunk_token[0] - test_case_1_single_data_chunk_token.squeeze()).numel()
        print(f"""Test case 1 passed! chunk mode and full mode are equal in {inference_mode} mode,
              diff token cnt in full: {diff_full_cnt}, diff token cnt in chunk: {diff_chunk_cnt}""")

    #Test case 2 is short audio
    test_case_2_single_data_chunk_token = test_case_2_single_data_chunk_token.squeeze()
    test_case_2_single_data_full_token = test_case_2_single_data_full_token.squeeze()

    valid_chunk_tokens_len = test_case_2_single_data_chunk_token.shape[0]
    padded_full_token_len = test_case_2_single_data_full_token.shape[0]

    valid_chunk_tokens = multiple_data_chunk_token[1][:valid_chunk_tokens_len]
    valid_full_tokens = multiple_data_full_token[1][:padded_full_token_len]
    # valid_chunk_tokens = padded_chunk_token
    # valid_full_tokens = padded_full_token


    if inference_mode == 'samplewise':
        assert torch.equal(test_case_2_single_data_chunk_token, valid_chunk_tokens)
        assert torch.equal(test_case_2_single_data_full_token, valid_full_tokens)
        print(f"Test case 2 passed! chunk mode and full mode are equal in {inference_mode} mode")
    elif inference_mode == 'batchwise':
        diff_full_cnt = torch.nonzero(valid_full_tokens - test_case_2_single_data_full_token).numel()
        diff_chunk_cnt = torch.nonzero(valid_chunk_tokens - test_case_2_single_data_chunk_token).numel()
        print(f"""Test case 2 passed! chunk mode and full mode are equal in {inference_mode} mode,
              diff token cnt in full: {diff_full_cnt}, diff token cnt in chunk: {diff_chunk_cnt}""")
        



if __name__ == "__main__":
    # Set up the argument parser
    import argparse
    parser = argparse.ArgumentParser(
        description="Run and compare chunk-wise vs. full inference for a UMM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )

    # Define command-line arguments
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
        help="Path to the model checkpoint file."
    )
    parser.add_argument(
        '--module',
        type=str,
        default="recipes.umm2.modules.stages.stage2.Stage2",
        help="The class path for the PyTorch Lightning module."
    )
    parser.add_argument(
        '--requires',
        type=str,
        nargs='+', # Accepts one or more arguments and puts them in a list
        default=['token', 'loss'],
        help="List of required outputs from the model (e.g., token loss tag)."
    )
    parser.add_argument(
        '--out_full',
        type=str,
        default="test_full.txt",
        help="Output file for tokens from full inference."
    )
    parser.add_argument(
        '--out_chunk',
        type=str,
        default="test_chunk.txt",
        help="Output file for tokens from chunk-wise inference."
    )
    parser.add_argument(
        '--token_rate',
        type=int,
        default=25,
        help="Token rate"
    )

    parser.add_argument(
        '--inference_mode',
        type=str,
        default='samplewise',
        choices=['samplewise', 'batchwise'],
        help='Choose the inference function to use: samplewise or batchwise.'
    )

    args = parser.parse_args()
    test_inference_consistency(**vars(args))

        

