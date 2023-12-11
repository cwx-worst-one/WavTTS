from recipes.mi1.models.music_sft import MI1_MusicClassificationMusicSFT
from recipes.mi1.models.tokenizers import UMMTokenizer


if __name__ == "__main__":


    model_mcc30k_genre = MI1_MusicClassificationMusicSFT.load_from_checkpoint("/mnt/bn/janne-research-xl/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_genre_zh_step=0030000.ckpt")
    model_mcc30k_genre = model_mcc30k_genre.to("cuda")
    print("Vocab:", model_mcc30k_genre.tag_tokenizer.vocab)

    model_vocal_gender = MI1_MusicClassificationMusicSFT.load_from_checkpoint("/mnt/bn/janne-research-xl/models/m1/m1_music_sft_vocal_v0_umm_1layer_classification_vocalgender_step=0030000.ckpt")
    model_vocal_gender = model_vocal_gender.to("cuda")

    print("Vocab:", model_vocal_gender.tag_tokenizer.vocab)

    model_mcc30k_mood = MI1_MusicClassificationMusicSFT.load_from_checkpoint("/mnt/bn/janne-research-xl/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_mood_zh_step=0030000.ckpt")
    model_mcc30k_mood = model_mcc30k_mood.to("cuda")

    print("Mood:", model_mcc30k_mood.tag_tokenizer.vocab)

    model_mcc30k_scene = MI1_MusicClassificationMusicSFT.load_from_checkpoint("/mnt/bn/janne-research-xl/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_scene_zh_step=0030000.ckpt")
    model_mcc30k_scene = model_mcc30k_scene.to("cuda")

    print("Scene:", model_mcc30k_mood.tag_tokenizer.vocab)

    audio_tokenizer = UMMTokenizer().to("cuda")


    audio = ... # [batch, channels, samples]
    hidden_states = audio_tokenizer(audio.to(audio_tokenizer.device)).hidden_states


    result_genres = model_mcc30k_genre.predict_tags(hidden_states)

    result_vocal_gender = model_vocal_gender.predict_tags(hidden_states)

    result_mood = model_mcc30k_mood.predict_tags(hidden_states)

    result_scene = model_mcc30k_scene.predict_tags(hidden_states)

    print("Predicted genre tag names:", result_genres.tag_names)
    print("Predicted vocal gender tag names:", result_vocal_gender.tag_names)
    print("Predicted mood tag names:", result_mood.tag_names)
    print("Predicted scene tag names:", result_scene.tag_names)
    
    # tag probabilities
    # print(result.tag_probabilities)