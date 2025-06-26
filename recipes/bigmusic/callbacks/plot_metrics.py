import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def replace_none_with_nan(obj):
    if isinstance(obj, list):
        return [replace_none_with_nan(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: replace_none_with_nan(v) for k, v in obj.items()}
    elif obj is None:
        return np.nan
    return obj

def plot_wer(path_result):

    path_result = str(path_result)
    wer = {}
    pinyin_wer = {}
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    categories = [x for x in categories if x != 'asr_timestamps']
    categories = [x for x in categories if x != 'deepchorus_result']
    wer['all'] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
    pinyin_wer['all'] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
    keys = ['wer', 'ins', 'subs', 'dels']

    for category in categories:
        wer[category] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
        pinyin_wer[category] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        for filename in filenames:
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            metadata = replace_none_with_nan(metadata)
            for key in keys:
                wer[category][key].append(metadata['wer'][key])
                wer[category]['badcase'] += metadata['wer']['badcase']
                pinyin_wer[category][key].append(metadata['pinyin_wer'][key])
                pinyin_wer[category]['badcase'] += metadata['pinyin_wer']['badcase']
                wer['all'][key].append(metadata['wer'][key])
                wer['all']['badcase'] += metadata['wer']['badcase']
                pinyin_wer['all'][key].append(metadata['pinyin_wer'][key])
                pinyin_wer['all']['badcase'] += metadata['pinyin_wer']['badcase']

        filename_category_metric = os.path.join(path_result, category, 'metrics.json')
        metric = json.load(open(filename_category_metric, 'r', encoding='utf-8'))
        metric = replace_none_with_nan(metric)
        wer[category]['merged'] = metric['wer']
        pinyin_wer[category]['merged'] = metric['pinyin_wer']
    filename_all_metric = os.path.join(path_result, 'all_metrics.json')
    metric = json.load(open(filename_all_metric, 'r', encoding='utf-8'))
    metric = replace_none_with_nan(metric)
    wer['all']['merged'] = metric['wer']
    pinyin_wer['all']['merged'] = metric['pinyin_wer']

    categories = ['all'] + categories
    ymax = 2.0
    color_median = [0, 0.5, 0]
    canvas_colors = [[254/255, 249/255, 231/255], [230/255, 237/255, 237/255]]
    fig, ax = plt.subplots(1, 2, figsize=(max(20, 2.5*len(categories)), 10))
    for idx_plot in range(2):
        if idx_plot == 0:
            wer_cur = wer
        if idx_plot == 1:
            wer_cur = pinyin_wer
        positions_global = []

        # color canvas
        for c, x in enumerate(np.arange(1, len(categories)*2+1, 2)):
            rectangle = patches.Rectangle((x, 0), 2, ymax, linewidth=0, facecolor=canvas_colors[c%2])
            ax[idx_plot].add_patch(rectangle)
        
        for ic, category in enumerate(categories):
            x = (ic + 1) * 2
            data = []
            positions = []
            for ik, key in enumerate(keys):
                positions.append(x + ik*0.3-0.45)
                data.append(wer_cur[category][key])
            boxprops = dict(facecolor='skyblue', color='grey')
            medianprops = dict(color=color_median, linewidth=2.5)
            flierprops = dict(marker='.', color='r', alpha=0.5)
            boxplot = ax[idx_plot].boxplot(
                data, 
                positions=positions, 
                patch_artist=True, 
                notch=True, 
                vert=True, 
                boxprops=boxprops, 
                medianprops=medianprops, 
                flierprops=flierprops,
                )
            for line in boxplot['medians']:
                x, y = line.get_xydata()[1]
                ax[idx_plot].plot([x-0.15, x+0.1], [y, y], color=color_median, linewidth=3)  # 调整中位数线的起点和终点

            colors = ['lightgreen', [0.9,1.0,0.95],[0.9,1.0,0.95],[0.9,1.0,0.95], ]
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
            for i, pos in enumerate(positions[:1]):
                ax[idx_plot].text(pos, 1.075 * ymax, '%.3f'%wer_cur[category]['merged']['wer'], 
                        horizontalalignment='center', fontsize=10, color='black')
                ax[idx_plot].text(pos, 1.05 * ymax, '%.3f'%np.mean(data[i]), 
                        horizontalalignment='center', fontsize=10, color='black')
                ax[idx_plot].text(pos, 1.025 * ymax, '%.3f'%np.median(data[i]), 
                        horizontalalignment='center', fontsize=10, color='black')
            ax[idx_plot].text(positions[3]+0.1, 0.83, '%d'%len([x for x in data[3] if x>0.8]), 
                    horizontalalignment='center', fontsize=12, color='red')
            
            positions_global.extend(positions)
            ax[idx_plot].text(x-0.5, -ymax/7, category, horizontalalignment='center', fontsize=12, rotation=10)
            ax[idx_plot].text(x-0.5, -ymax/7-0.06, '(%d)'%len(wer_cur[category]['wer']), horizontalalignment='center', fontsize=10)
        
        ax[idx_plot].text(0, 1.075 * ymax, 'merged:')
        ax[idx_plot].text(0, 1.05 * ymax, 'mean:')
        ax[idx_plot].text(0, 1.025 * ymax, 'median:')
        ax[idx_plot].text(0.8, 0.83, 'badcase', color='red')
        ax[idx_plot].axhline(y=0.8, color=[1,0.7,0.7], linestyle=':', linewidth=2)

        ax[idx_plot].set_xticks(positions_global, keys * len(categories), rotation=-90)
        ax[idx_plot].set_ylim([0, ymax])
        ax[idx_plot].set_xlim( [ 1, max(10, len(categories)*2+1) ] )
        ax[idx_plot].set_yticks(np.arange(0, ymax+0.01, 0.2))
        ax[idx_plot].grid(True, axis='y')

        for ic, category in enumerate(categories):
            x = (ic + 1) * 2
            ax[idx_plot].text(0.8, -ymax/6 - ymax/25, 'wer', horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(x, -ymax/6 - ymax/25, '%.3f'%wer_cur[category]['merged']['wer'], horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(0.8, -ymax/6 - ymax/25*2, 'ins', horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(x, -ymax/6 - ymax/25*2, '%.3f'%wer_cur[category]['merged']['ins'], horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(0.8, -ymax/6 - ymax/25*3, 'subs', horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(x, -ymax/6 - ymax/25*3, '%.3f'%wer_cur[category]['merged']['subs'], horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(0.8, -ymax/6 - ymax/25*4, 'dels', horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(x, -ymax/6 - ymax/25*4, '%.3f'%wer_cur[category]['merged']['dels'], horizontalalignment='center', fontsize=12, color='black')
            ax[idx_plot].text(0.8, -ymax/6 - ymax/25*5, 'badcase', horizontalalignment='center', fontsize=12, color='black')
            tmp = 'nan' if np.isnan(wer_cur[category]['merged']['badcases']) else '%d'%wer_cur[category]['merged']['badcases']
            ax[idx_plot].text(x, -ymax/6 - ymax/25*5, tmp, horizontalalignment='center', fontsize=12, color='black')
        
    ax[0].set_position([0.06, 0.25, 0.4, 0.6])
    ax[1].set_position([0.55, 0.25, 0.4, 0.6])
    ax[0].text(len(categories)+1, 1.15 * ymax, 'wer', horizontalalignment='center', fontsize=14, color='black')
    ax[1].text(len(categories)+1, 1.15 * ymax, 'pinyin_wer', horizontalalignment='center', fontsize=14, color='black')

    filename_result = os.path.join(path_result, 'wer.png')
    fig.savefig(filename_result)
    print ('wer metrics figure has been saved as: ', filename_result)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path_result', type=str, help='')
    args = parser.parse_args()
    path_result = args.path_result

    plot_wer(path_result)