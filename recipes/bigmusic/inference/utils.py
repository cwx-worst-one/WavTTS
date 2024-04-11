import os


def get_sami_tokenizer():
    from recipes.datasets.mcc.sami_tokenizer import SamiTokenizer
    from sami_tts_api.sail import download_model
    fe_version="42.0"
    fe_task="tts_chinese_frontend_model"
    model_dir = os.path.join(
        os.environ['AI_MUSIC_DIR'],
        '../../dump/ai_music/20240223.svs.dump'
    )
    fe = download_model(fe_task, model_dir, fe_version)

    return SamiTokenizer(fe=fe)


def download_checkpoint_from_hdfs(hdfs_path, output_path):
    if not os.path.exists(output_path):
        os.system(f'hdfs dfs -get {hdfs_path} {output_path}')
    assert os.path.exists(output_path), "Model download failed"
