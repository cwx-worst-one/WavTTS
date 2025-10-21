import os
import re
import csv
import glob
import json
import pandas as pd


import os
import json
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report



### Merge Functions
def merge_results_from_ranks(
        results_dir: str, 
        pattern: str = "results_tagging_rank_*.json"):
    """Merge results from all rank files in the directory."""
    merged_data = {}

    # Find all result files matching the pattern
    pattern_path = os.path.join(results_dir, pattern)
    files = glob.glob(pattern_path)

    if not files:
        print(f"No files found matching pattern: {pattern_path}")
        return []

    print(f"Found {len(files)} rank files: {[os.path.basename(f) for f in files]}")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract rank number from filename
            rank_match = re.search(r'rank_(\d+)', file_path)
            rank = int(rank_match.group(1)) if rank_match else 0

            print(f"Processing rank {rank} with {len(data)} samples")

            for item in data:
                uttid = item.get('uttid')
                if uttid:
                    if uttid not in merged_data:
                        merged_data[uttid] = item
                        merged_data[uttid]['rank'] = rank
                    else:
                        # Keep the one with lower rank (assuming lower rank is better)
                        if rank < merged_data[uttid].get('rank', float('inf')):
                            merged_data[uttid] = item
                            merged_data[uttid]['rank'] = rank

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return merged_data

def merge_result_data(
        results_dir: str, 
        dset_id: str):
    pattern = "results_tagging_rank_*.json"
    merged_data = merge_results_from_ranks(results_dir, pattern)

    with open('results_merged.json', 'w') as f:
        json.dump(merged_data, f)


def compile_pred_dataset(outdir, dset_id):
    merge_result_data(
        results_dir='results', 
        dset_id=dset_id)
    with open('results_merged.json', 'r') as f:
        pred_data = json.load(f)
    print('[i] dset_id:', dset_id)
    print('[i]     num:', len(pred_data))

    rows = []
    for uttid, v in pred_data.items():
        row = {
            'uttid': uttid,
            'logits_Y': v['scores'][0],
            'logits_N': v['scores'][1],
            'rating_pred': v['predict_results'][0],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(outdir, f'score-{dset_id}_pred.csv'), index=False)


def compile_anno_dataset(outdir, dset_id):
    merge_result_data(
        results_dir='results', 
        dset_id=dset_id)
    with open('results_merged.json', 'r') as f:
        pred_data = json.load(f)
    print('[i] dset_id:', dset_id)
    print('[i]     num:', len(pred_data))

    rows = []
    for uttid, v in pred_data.items():
        # import pdb; pdb.set_trace()

        if isinstance(v['raw']['music_rating'], str):
            rating = int(v['raw']['music_rating'])
        elif isinstance(v['raw']['music_rating'], list):
            rating = int(v['raw']['music_rating'][0])
        else:
            rating = int(v['raw']['music_rating'])

        row = {
            'uttid': uttid,
            'rating_anno': rating,
            'url': v['raw']['audio'],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f'score-{dset_id}_anno.csv'), index=False)

if __name__ == '__main__':
    dset_id = os.getenv('DATASET_ID', default='')
    # dset_id = '26140'
    # dset_id = 20821
    # dset_id = "25442n25443"
    # dset_id = "25444"
    outdir = 'score_stats'
    os.makedirs(outdir, exist_ok=True)
    compile_pred_dataset(outdir, dset_id)
    compile_anno_dataset(outdir, dset_id)
    
    # merge pred and anno for calibration & evaluation
    df_anno = pd.read_csv(os.path.join(outdir, f'score-{dset_id}_anno.csv'))
    df_pred = pd.read_csv(os.path.join(outdir, f'score-{dset_id}_pred.csv'))
    df_merged = pd.merge(df_pred, df_anno, on='uttid', how='inner')
    df_merged.to_csv(os.path.join(outdir, f'score-{dset_id}_merged.csv'), index=False) # save merged one to local

    ############################
    threshold = 12.0
    outdir = 'score_stats'
    os.makedirs(outdir, exist_ok=True)
    # dset_id = '25444'
    # dset_id = "25442n25443"
    # dset_id = "20821"

    dset_id = os.getenv('DATASET_ID', default='')
    df_merged = pd.read_csv(os.path.join(outdir, f'score-{dset_id}_merged.csv'))

    #######
    y_pred = []
    y_true = []
    for idx, row in df_merged.iterrows():
        logits_Y = row['logits_Y']
        logits_N = row['logits_N']
        rating = int(row['rating_anno'])

        quality_pred = 1 if logits_Y >= threshold else 0
        quality_anno = 1 if rating > 3 else 0
        
        y_pred.append(quality_pred)
        y_true.append(quality_anno)
        # print(row)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Classification Report (Precision, Recall, F1, Support)
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    print(report )

    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix, threshold {threshold:.2f}')
    # plt.show()
    plt.savefig(os.path.join(outdir, f'score_cm_{dset_id}.png'))

    with open(os.path.join(outdir, f"score_{dset_id}_report.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))  # converts numpy array to string
        f.write("\n\nClassification Report:\n")
        f.write(report)