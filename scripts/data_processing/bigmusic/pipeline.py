import functools
import io
import itertools
import os
import random

import click
import numpy as np
import psutil
import ray
import schedule
from bigspeech_process.controller import Controller
from bigspeech_process.data.webdataset import IndexReader, IndexWriter, WebdatasetWriter
from bigspeech_process.operators import sami
from bigspeech_process.operators.mss.predicator import \
    Predictor as MSSPredictor
from bigspeech_process.operators.tts.frontend import SamiTTSFrontendPredictor
from bigspeech_process.util.decorator import retry
from bigspeech_process.util.log import getlogger
from bigspeech_process.util.wave_utils import normalize_audio
from pyarrow import fs
from samantha.dataio.webdataset.extension import IndexedWebDataset
from scipy.io import wavfile
from tqdm import tqdm
from ray.util import ActorPool

@retry(
    retry_times=3, interval=3,
    exc_cls=(sami.exception.RPCTimeout, sami.exception.UnifiedFrontendError),
    return_value_instead_raise=(None, None, None)
)
@retry(retry_times=2, interval=3)
@retry(retry_times=-1, interval=3, exc_cls=(sami.exception.ConcurrentLimitError,))
@retry(retry_times=0, exc_cls=(sami.exception.ExceededDurationError, sami.exception.InvalidData), return_value_instead_raise=(None, None, None))
@retry(retry_times=30, interval=10, exc_cls=(ray.exceptions.RayActorError, ))
def processed_by_sami(local_server, npy):
    buf = io.BytesIO()
    wavfile.write(buf, 24000, npy)
    wav_bytes = buf.getvalue()

    audio_metric_ref = local_server.get_metric_by_audio.remote(wav_bytes)
    vad_ref = local_server.get_vad_by_audio.remote(wav_bytes)
    genre34_ref = local_server.get_genre34_by_audio.remote(wav_bytes)

    audio_metric, vad, genre34 = ray.get([audio_metric_ref, vad_ref, genre34_ref])
    return audio_metric, vad, genre34

def process_npy_to_16kwav(npy):
    wavform = io.BytesIO()
    wavfile.write(wavform, 24000, npy)
    audio_bytes = normalize_audio(audio_bytes=wavform.getvalue(), sample_rate=16000)
    return audio_bytes

@ray.remote
def async_lyric_recognize(batch, pool):
    from bigspeech_process.operators.infer import sa_asr_offline
    wav16k = [process_npy_to_16kwav(item['_npy']) for item in batch]
    return sa_asr_offline.asr_recoginize_batched(wav16k, pool)
    # return list(pool.map(lambda a, v: sa_asr_offline.asr_recoginize_batched(v, a), [wav16k]))[0]

@ray.remote
@retry(retry_times=30, interval=10, exc_cls=(ray.exceptions.RayActorError,))
def async_phoneme_recognize(batch, pool):
    result = []
    for item in batch:
        lyrics = item['__index_data__']['lyrics']
        for lyric in lyrics['result']:
            text_list = [utterances['text'] for utterances in lyric['utterances']]
            for idx, (phoneme, tn_text) in enumerate(list(pool.map(lambda a, v: a.predict.remote(v), [text_list]))[0]):
                lyric['utterances'][idx]['phoneme'] = phoneme
                lyric['utterances'][idx]['normalized_text'] = tn_text
        result.append({'lyrics': lyrics})
    return result

@ray.remote
def async_sami_recognize(batch, local_service_mgr):
    result = []
    for item in batch:
        actor = local_service_mgr.get_one()
        metric, vad, genre34 = processed_by_sami(actor, item['_npy'])
        result.append({
            'audiometrics': metric,
            'vad': vad,
            'genre34': genre34
        })
    return result

@ray.remote
@retry(retry_times=30, interval=10, exc_cls=(ray.exceptions.RayActorError, ray.exceptions.ObjectLostError, ray.exceptions.GetTimeoutError))
def submit_and_get(actors, timeout, op, *args, **kwargs):
    actor = random.sample(actors, 1)[0]
    return ray.get(op(actor, *args, **kwargs), timeout=timeout)

@ray.remote
def async_mss(batch, actor_list):
    futures = []
    for item in batch:
        future = submit_and_get.remote(actor_list, 30, lambda a, v: a.predict.remote(v), item['_npy'])
        futures.append(future)

    return ray.get(futures)

def fetch_batch(dataset, batch_size, filter_fn=None):
    cnt = 0
    batch = []
    if filter_fn:
        dataset = filter(filter_fn, dataset)
    for item in dataset:
        batch.append(item)
        cnt += 1
        if cnt >= batch_size:
            yield batch
            batch = []
            cnt = 0
    if batch:
        yield batch

def pipeline(input_file, output_dir, batch_size, index_file, filesystem, labels, actor_map):
    logger = getlogger()
    index_ds = IndexReader(filesystem.normalize_path(index_file), filesystem=filesystem)
    progress = tqdm(total=index_ds.count(), bar_format='Progress: {n}/{total}, Rate: {rate_fmt}')
    pid = os.getpid()
    psproc = psutil.Process(pid)
    schedule.every(10).seconds.do(lambda: logger.info('[PID=%d]memory: %s', pid, psproc.memory_full_info()))
    index_ds.close()
    labels = labels.split(',')

    dataset = IndexedWebDataset({input_file: index_file})
    print("index pppath", os.path.join(filesystem.normalize_path(output_dir), os.path.basename(input_file).replace('.tar', '.idx')))
    index_writer = IndexWriter(os.path.join(filesystem.normalize_path(output_dir), os.path.basename(input_file).replace('.tar', '.idx')), filesystem=filesystem)
    print("tar pppath", os.path.join(output_dir, f'mss_{os.path.basename(input_file)}'))
    mss_tar_writer = WebdatasetWriter(os.path.join(output_dir, f'mss_{os.path.basename(input_file)}'), filesystem=filesystem)

    for batch in fetch_batch(dataset, batch_size):
        if not batch:
            continue

        for item in batch:
            npy = np.frombuffer(item['audio.npy'], np.int16)
            if len(npy.shape) == 2:
                npy = npy[0]
            item['_npy'] = npy

        future_map = {}
        for label in labels:
            if label not in actor_map:
                continue
            actor = actor_map[label]
            future_map[label] = actor(batch)

        result_map = {}
        for label, future in future_map.items():
            result_map[label] = ray.get(future)

        for idx, item in enumerate(batch):
            metadata = item['__index_data__']
            for label in labels:
                if label not in result_map:
                    continue
                if label == 'mss':
                    vocal, acc = result_map[label][idx]
                    mss_tar_writer.write(item['__key__'], item['audio.npy'], **{'vocal.npy': vocal, 'acc.npy': acc})
                else:
                    metadata.update(result_map[label][idx])
            index_writer.write(item['__key__'], metadata)

        progress.update(len(batch))
        schedule.run_pending()

    index_writer.close()
    mss_tar_writer.close()

@click.option('--index2url', required=True, type=str, help='a dir include multiple index2url.txt files or a single index2url.txt file')
@click.option('--max_concurrency', required=True, type=int, default=1)
@click.option('--labels', required=True, type=str, default='')
@click.option('--num_phoneme_actors', required=False, type=int, default=0)
@click.option('--num_sami_server', required=False, type=int, default=0)
@click.option('--num_mss_actors', required=False, type=int, default=0)
@click.option('--num_asr_actors', required=False, type=int, default=0)
@click.option('--asr_model_name', required=False, type=str, default='en_us_lyric')
@click.option('--output_dir', required=True, type=str)
@click.option('--batch_size', required=False, type=int, default=16)
@click.command()
def main(output_dir, asr_model_name, num_asr_actors, num_mss_actors, num_phoneme_actors, labels, max_concurrency, index2url, num_sami_server, batch_size):
    if not output_dir.startswith('hdfs://'):
        raise Exception("output dir must start with 'hdfs://'")

    hdfs_hostname = 'hdfs://' + output_dir.replace('hdfs://', '').split('/')[0]
    output_dir = output_dir.replace(hdfs_hostname, '')
    index2url = index2url.replace(hdfs_hostname, '')

    with Controller(
        max_concurrency=max_concurrency,
        ckpt_init_args={
            'ckpt_dir': f'{output_dir}/ckpt/'},
    ) as ctl:
        filesystem, _ = fs.FileSystem.from_uri(hdfs_hostname)
        ctl.ckpt.filesystem = filesystem

        # ASR actor
        asr_actors_list = []
        if 'lyric' in labels:
            from bigspeech_process.operators.infer import sa_asr_offline
            ASROfflineExecuteActor = ray.remote(
                num_gpus=1, num_cpus=10, memory=25 << 30,
                concurrency_groups={"submit_api": 1000, "query_api": 1000}
            )(sa_asr_offline.ASROfflineExecute).options(max_restarts=-1)
            for _ in range(num_asr_actors):
                asr_actor = ASROfflineExecuteActor.remote(model_name=asr_model_name)  # en_us_lyric
                asr_actors_list.append(asr_actor)
        asr_actors = itertools.cycle(asr_actors_list)

        # TTS Frontend actor
        tts_frontend_predictors = []
        for _ in range(num_phoneme_actors):
            SamiTTSFrontendActor = ray.remote(
                num_cpus=2, memory=4 << 30,
                runtime_env={'env_vars': {'LD_LIBRARY_PATH': '/opt/tiger/sami_tts/libs'}}
            )(SamiTTSFrontendPredictor).options(max_restarts=-1)
            tts_frontend_predictors.append(SamiTTSFrontendActor.remote())
        tts_actor_pools = ActorPool(tts_frontend_predictors)

        # SAMI local server actor, for audiometrics, vad, genre34
        local_sami_server_manager = sami.SAMILocalServerManager()
        for _ in range(num_sami_server):
            local_sami_server_manager.start_new_sami_local_server()

        # MSS actor
        MSSActor = ray.remote(num_gpus=1, num_cpus=10, memory=15 << 30)(MSSPredictor).options(max_restarts=-1)
        mss_predictors = []
        for _ in range(num_mss_actors):
            mss_predictors.append(MSSActor.remote())
        # mss_actors = itertools.cycle(mss_predictors)
        # mss_actor_pools = ActorPool(mss_predictors)

        def args_generator():
            nonlocal output_dir
            index2url_txt_files = []
            index2url_fileinfo = filesystem.get_file_info(index2url)
            if index2url_fileinfo.type == fs.FileType.Directory:
                for file in filesystem.get_file_info(fs.FileSelector(index2url)):
                    if file.path.endswith('.txt') and file.type == fs.FileType.File:
                        index2url_txt_files.append(file.path)
            elif index2url_fileinfo.type == fs.FileType.File:
                index2url_txt_files.append(index2url)
            else:
                raise Exception("%s is %s" % (index2url, index2url_fileinfo.type))

            for txt_file in index2url_txt_files:
                if len(index2url_txt_files) > 1:
                    output_dir_name = os.path.basename(txt_file).removesuffix('.txt')
                    output_dir1 = os.path.join(output_dir, output_dir_name)
                    os.system(f"hdfs dfs -mkdir -p {output_dir1}")

                with filesystem.open_input_stream(txt_file) as fp:
                    for line in fp.read().decode().split('\n'):
                        if not line:
                            continue
                        tar_file, index_file = line.split('\t')
                        yield tar_file, output_dir1, batch_size, index_file, filesystem, labels, {
                            'tts_frontend': functools.partial(async_phoneme_recognize.remote, pool=tts_actor_pools) if tts_frontend_predictors else None,
                            'lyrics': functools.partial(async_lyric_recognize.remote, pool=next(asr_actors)) if asr_actors_list else None,
                            'mss': functools.partial(async_mss.remote, actor_list=mss_predictors) if mss_predictors else None,
                            'audiometrics_vad_genre34': functools.partial(async_sami_recognize.remote, local_service_mgr=local_sami_server_manager),
                        }
                    # break

        ctl.run(pipeline, args_generator)
        ctl.waitall()
        ray.get([asr_actor.stop.remote() for asr_actor in asr_actors_list])
        local_sami_server_manager.stop_all()

if __name__ == '__main__':
    main()


