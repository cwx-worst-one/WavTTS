import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from prettytable import PrettyTable

import euler
import thriftpy2


thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "services/idl/music_tagging.thrift"),
    "music_tagging_thrift"
)

from music_tagging_thrift import MusicTagging, TaggingRequest


def to_tos(local_path, prefix, tos_path="/opt/tiger/samantha/1.0.0.20/toscli", bucket="sa-music-model-zoo", ak="9NC3OBANMDH4TTPTO52E"):
    os.system(f'{tos_path} -bucket {bucket} -accessKey {ak} put -prefix {prefix} {local_path}')
    file_name = os.path.basename(local_path)
    url = f'https://tosv.byted.org/obj/sa-music-model-zoo/{prefix}{file_name}'
    # print(url)
    return url


class MusicMetric():
    def __init__(self, key, prompt_file, metric_name='acc') -> None:
        self.key = key
        self.generated_key = 'generated_' + key
        self.prompt_file = prompt_file
        self.metric_name = metric_name
        self.df = None
        self.setup()
    
    def setup(self):
        pass
    
    def metric_rule_func(self, x, y):
        return x == y

    def metric(self):
        return
    
    def report(self):
        pass


class TaggingMetric(MusicMetric):
    def __init__(self, tag_name, testset,  metric_name='acc') -> None:
        super().__init__(tag_name, testset, metric_name)     

    def setup(self):
        # self.df = pd.read_csv(self.prompt_file)[['index', 'prompt_category', 'prompt', 'genre', 'is_sinking', 'n_lines', 'structure_type', 'lyrics', 'predicted_mood', 'text_prompt', 'url']]
        self.time = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        self.df = pd.read_csv(self.prompt_file)[['index', self.key]]
        target = 'sd://lab.speech.music_tagging?cluster=default'
        self.client = euler.Client(MusicTagging,target=f'{target}&idc=lf', timeout=1200)

    def metric(self, df, round_float=4):
        if self.metric_name == 'acc':
            correct = df[df[self.rlt_key] == True].shape[0]
            total = df.shape[0]
            return (correct, total, round(correct / total, round_float))
        else:
            raise Exception(f'metric {self.metric_name} not implemented')

    def get_tag(self, url):
        answer = self.client.TaggingGenre20(TaggingRequest(track_id="test", url=url))
        result = eval(answer.result_json)['Genre20']['result']
        if len(result) == 0:
            print("not hit genre threshold")
            return "Other"
        if len(result) > 1:
            print(f"multiple rlts: {result}")
        return result[0]
    
    def process(self, output_dir, tos_path):
        generated_tags = []
        for row in self.df.itertuples():
            index = row.index
            generated_path = os.path.join(output_dir, index + ".generated.wav")
            tos_prefix = 'tmp/eval/wavs_test/' + self.time + '/'
            # file_name = os.path.basename(generated_path)
            # url = f'https://tosv.byted.org/obj/sa-music-model-zoo/tmp/eval/wavs_test/03-26-2024-00:04:59/{file_name}'

            url = to_tos(generated_path, tos_prefix, tos_path)
            
            predicted_tag = self.get_tag(url)
            
            generated_tags.append(predicted_tag)
        
        self.df[self.generated_key] = generated_tags

    def report(self, output_dir, tos_path):
        self.process(output_dir, tos_path)

        self.rlt_key = 'equal_' + self.key
        self.df[self.rlt_key] = self.df.apply(lambda x: self.metric_rule_func(x[self.key], x[self.generated_key]), axis=1)
        _, _, total_score = self.metric(self.df)
        print(f'{self.key} {self.metric_name}: {total_score}')
        
        report_tab = PrettyTable([self.key, self.metric_name, "support"])
        groups = self.df.groupby(self.key)
        sub_metric = {}
        for k, vdf in groups:
            k_correct, k_total, k_score = self.metric(vdf)
            report_tab.add_row([k, k_score, vdf.shape[0]])
            sub_metric[k] = (k_correct, k_total, k_score)
        report_tab.add_row(['total', total_score, self.df.shape[0]])
        
        print(report_tab)
        return {"total": total_score, "sub": sub_metric}


if __name__ == "__main__":
    # single testset usage:
    # python3 genre_metric.py --audio_dir <generated_audio_dir> --testset "<testset_path>" 
   
    # multiple testsets usage:
    # python3 genre_metric.py --audio_dir <generated_audio_dir1,dir2,dir3> --testset "<testset_path1,path2,path3>" --multi

    # example
    # python3 metric.py --audio_dir "/mnt/bn/bigmusic-lf/user/zh/samantha/assets/_tdiff/zh_test_1min_10/val,/mnt/bn/bigmusic-lf/user/zh/samantha/assets/_tdiff/zh_val_1min/val" --testset "/mnt/bn/bigmusic-lf/user/zh/data/cn_vocal/infer/1min/zh_test_1min_10_with_genre.csv,/mnt/bn/bigmusic-lf/user/zh/data/cn_vocal/infer/1min/zh_val_1min_with_genre.csv" --multi 

    parser = argparse.ArgumentParser(description='genre metric')
    # multiple audio_dir and testset should be aligned
    parser.add_argument('--audio_dir', type=str, required=True, help='generated audios directory, multiple directories seperated by ,')
    parser.add_argument('--testset', type=str, required=True, default='/mnt/bn/bigmusic-lf/user/zh/data/cn_vocal/infer/1min/douyin0314_12.csv', help='testset path, multiple testsets seperated by ,')
    parser.add_argument('--tos', type=str, default="/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli", help="toscli path")
    parser.add_argument('--multi', action='store_true', help='multiple testsets')

    args = parser.parse_args()
    if not args.multi:
        metric = TaggingMetric("genre", args.testset)
        scores = metric.report(args.audio_dir, args.tos)
    else:
        dirs = args.audio_dir.split(",")
        testsets = args.testset.split(",")
        total_scores = []
        for d, t in zip(dirs, testsets):
            metric = TaggingMetric("genre", t)
            scores = metric.report(d, args.tos)
            total_scores.append(scores)
        
        summary = {}
        total = 0
        for score in total_scores:
            for k,v in score['sub'].items():
                if k not in summary:
                    summary[k] = {"correct": v[0], "total": v[1]}
                else:
                    summary[k] = {"correct": summary[k]["correct"] + v[0], "total": summary[k]["total"] + v[1]}
        
        print("genre metric summary:")
        report_tab = PrettyTable(["genre", "acc", "support"])
        t_c, t_n = 0, 0
        for k, v in summary.items():
            t_c += v["correct"]
            t_n += v["total"]
            report_tab.add_row([k, round(v["correct"] / v["total"], 4), v["total"]])
        report_tab.add_row(["total", round(t_c / t_n, 4), t_n])
        print(report_tab)
