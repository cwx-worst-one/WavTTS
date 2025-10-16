import os
import re
import csv
import glob
import json
import pprint
import pandas as pd

from rouge_score import rouge_scorer

DATASET_ID = "24336"

def _parse_gemini_v2(gemini_v2: dict) -> dict:
    if not gemini_v2:
        return {}
    gemini_meta = {}
    gemini_meta['genre'] = gemini_v2.get('genre', {}).get('primary', [])
    gemini_meta['genre_extra'] = gemini_v2.get('genre', {}).get('additional', [])
    gemini_meta['mood'] = gemini_v2.get('mood', {}).get('keywords', [])
    gemini_meta['language'] = gemini_v2.get('language', [])
    gemini_meta['tempo'] = \
        gemini_v2.get('musical_features', {}).get("tempo_keywords", {}).get("tempo", []) + \
        gemini_v2.get('musical_features', {}).get("tempo_keywords", {}).get("BPM", [])
    try:
        gemini_meta['gender'] = \
            [gender for keyword in gemini_v2.get('musical_features', {}).get("vocal", {}).get("keywords", []) for gender in keyword.get("gender", [])]
    except:
        pass
    gemini_meta['scene'] = gemini_v2.get('scene', {}).get("keywords", [])
    scene_phrases = gemini_v2.get('scene', {}).get("phrases", [])
    # remove tailing .
    scene_phrases = [phrase.rstrip('.') for phrase in scene_phrases]
    gemini_meta['scene_phrase'] = scene_phrases
    gemini_meta['timbre'] = \
        [timbre for keyword in gemini_v2.get('musical_features', {}).get("vocal", {}).get("keywords", []) for timbre in keyword.get("timbre", [])]
    gemini_meta['instrument'] = \
        gemini_v2.get('musical_features', {}).get("instruments", {}).get("keywords", [])
    gemini_meta['era'] = gemini_v2.get('additional', {}).get("era_style", [])
    arrangement_keywords = gemini_v2.get('musical_features', {}).get("arrangement", {}).get("keywords", [])
    # gemini_meta['arrangement'] = arrangement_keywords
    gemini_meta['melody'] = gemini_v2.get('musical_features', {}).get("melody", {}).get("keywords", [])
    gemini_meta['rhythm'] = gemini_v2.get('musical_features', {}).get("rhythm", {}).get("keywords", [])
    gemini_meta['key'] = gemini_v2.get('musical_features', {}).get("key", [])
    gemini_meta['imagery'] = gemini_v2.get('abstract_descriptors', {}).get("imagery", [])
    gemini_meta['synesthesia_tags'] = gemini_v2.get('abstract_descriptors', {}).get("synesthesia_tags", [])
    gemini_meta['vibe'] = gemini_v2.get('abstract_descriptors', {}).get("vibe", [])
    gemini_meta['audio_features'] = gemini_v2.get('audio_features', {}).get("audio_feature_keywords", [])   
    gemini_meta['additional'] = gemini_v2.get('additional', {}).get("features", [])
    gemini_meta['description'] = gemini_v2.get('description', {}).get('global_description', [])
    gemini_meta['description_long'] = gemini_v2.get('description', {}).get('global_description_long', [])
    return gemini_meta

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

def merge_result_data(input_dir):
    pattern = "results_tagging_rank_*.json"
    merged_data = merge_results_from_ranks(input_dir, pattern)

    with open('results_merged.json', 'w') as f:
        json.dump(merged_data, f)

##########################################################################################3
def metric_set_rouge_score(pred_list, gt_list):
    # sort and string handling
    pred_list = [t.replace('-', ' ').lower() for t in pred_list]
    gt_list = [t.replace('-', ' ').lower() for t in gt_list]
    pred_text = ';'.join(pred_list).replace('-', ' ')
    gt_text = ';'.join(gt_list).replace('-', ' ')

    # score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt_text, pred_text)

    rouge1 = scores['rouge1'].fmeasure
    rougeL = scores['rougeL'].fmeasure
    return rouge1, rougeL

def metric_rouge_score(pred_text, gt_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt_text, pred_text)

    # print(scores)
    rouge1 = scores['rouge1'].fmeasure
    rougeL = scores['rougeL'].fmeasure
    return rouge1, rougeL

def metric_set_f1(pred_tags, anno_tags):
    '''pred_tags (List), anno_tags (List)'''
    # string handling
    pred_tags = [t.replace('-', ' ').lower() for t in pred_tags]
    anno_tags = [t.replace('-', ' ').lower() for t in anno_tags]
    
    # evaluation
    pred_set = set(pred_tags)
    anno_set = set(anno_tags)
    tp = len(pred_set & anno_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(anno_set) if anno_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall

def metric_abs(pred, anno):
    return abs(float(pred) - float(anno))

def parse_pred_result(pred_text):
    # # pattern = r"This audio's ([^ ]+) is (.*?)(?=This audio's|$)"
    # pattern = r"\[(\w+):\s*([^\]]+)\]"
    # matches = re.findall(pattern, pred_text, re.DOTALL)
    # pred_data_song = {}
    # for key, value in matches:
    #     value = value.strip().rstrip(".")
    #     if key not in ["description", "description_long"]:
    #         value = [v.strip() for v in value.split(",")]
    #     else:
    #         value = [value]
    #     pred_data_song[key] = value
    pred_data_song = json.loads(pred_text)
    return pred_data_song

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def evaluate(pred_data_song, anno_data_song):
    '''pred_data_song (Dict), anno_data_song (Dict)'''
    eval_keys = [
        'tempo', 'gender', 'scene', 'scene_phrase', 'timbre', 'instrument', 'era', 'melody', 'rhythm', 'key', 'imagery', 'synesthesia_tags', 'vibe', 'audio_features','additional', 'description', 'description_long']
    
    # genre
    _, genre_rougeL = metric_set_rouge_score(
        pred_data_song.get('genre', []), anno_data_song['genre'])

    # genre_extra
    genre_extra_f1, _, _ = metric_set_f1(
        pred_data_song.get('genre_extra', []), anno_data_song['genre_extra'])
    _, genre_extra_rougeL = metric_set_rouge_score(
        pred_data_song.get('genre_extra', []), anno_data_song['genre_extra'])

    # mood
    _, mood_rougeL = metric_set_rouge_score(
        pred_data_song.get('mood', []), anno_data_song['mood'])

    # language
    language_f1, _, _ = metric_set_f1(
        pred_data_song.get('language', []), anno_data_song['language'])

    #### rhythm
    pred_data_song['tempo'] = sorted(pred_data_song.get('tempo', ["0.0", ""]), key=lambda x: not is_number(x))
    anno_data_song['tempo'] = sorted(anno_data_song.get('tempo', ["0.0", ""]), key=lambda x: not is_number(x))
    # print(pred_data_song['tempo'])
    # print(anno_data_song['tempo'])
    # bpm
    bpm_pred = pred_data_song['tempo'][0]
    bpm_anno = anno_data_song['tempo'][0]
    try:
        tempo_abs = metric_abs(bpm_pred, bpm_anno)
    except:
        tempo_abs = -1

    # tempo
    tempo_pred = ''
    tempo_anno = ''
    try:
        tempo_pred = pred_data_song['tempo'][1]
        tempo_anno = anno_data_song['tempo'][1]
   
        tempo_f1, _, _ = metric_set_f1(tempo_pred, tempo_anno)
    except:
        tempo_f1 = -1

    # gender
    gender_f1, _, _ = metric_set_f1(pred_data_song.get('gender', []),  anno_data_song['gender'])

    # scene
    _, scene_rougeL = metric_set_rouge_score(
        pred_data_song.get('scene', []), anno_data_song['scene'])

    # scene phrase
    _, scene_phrase_rougeL = metric_set_rouge_score(
        pred_data_song.get('scene_phrase', []), anno_data_song['scene_phrase'])

    # timbre
    timbre_f1, _, _ = metric_set_f1(
        pred_data_song.get('timbre', []),  anno_data_song['timbre'])
    _, timbre_rougeL = metric_set_rouge_score(
        pred_data_song.get('timbre', []), anno_data_song['timbre'])

    # instrument
    instrument_f1, instrument_precision, instrument_recall = metric_set_f1(
        pred_data_song.get('instrument', []),  anno_data_song['instrument'])
    _, instrument_rougeL = metric_set_rouge_score(
        pred_data_song.get('instrument', []), anno_data_song['instrument'])

    # era
    era_f1, _, _ = metric_set_f1(
        pred_data_song.get('era', []), anno_data_song['era'])
    
    # melody
    _, melody_rougeL = metric_set_rouge_score(
        pred_data_song.get('melody', []), anno_data_song['melody'])

    # rhythm
    _, rhythm_rougeL = metric_set_rouge_score(
        pred_data_song.get('rhythm', []), anno_data_song['rhythm'])

    # key
    key_f1, _, _ = metric_set_f1(
        pred_data_song.get('key', []), anno_data_song['key'])

    # imagery
    _, imagery_rougeL = metric_set_rouge_score(
        pred_data_song.get('imagery', []), anno_data_song['imagery'])

    # synesthesia tags
    _, synesthesia_tags_rougeL = metric_set_rouge_score(
        pred_data_song.get('synesthesia_tags', []), anno_data_song['synesthesia_tags'])

    # vibe
    _, vibe_rougeL = metric_set_rouge_score(
        pred_data_song.get('vibe', []), anno_data_song['vibe'])

    # audio features
    audio_features_f1, _, _ = metric_set_f1(
        pred_data_song.get('audio_features', []), anno_data_song['audio_features'])

    # additional
    _, additional_rougeL = metric_set_rouge_score(
        pred_data_song.get('additional', []), anno_data_song['additional'])

    # description
    description = pred_data_song.get('description', [''])[0]
    _, description_rougeL = metric_rouge_score(
        description, anno_data_song['description'][0])

    # description
    description_long = pred_data_song.get('description_long', [''])[0]
    _, description_long_rougeL = metric_rouge_score(
        description_long, anno_data_song['description_long'][0])

    # import pdb; pdb.set_trace()
    # result
    result = {
        # genre
        'genre_pred': pred_data_song.get('genre', []),
        'genre_anno': anno_data_song['genre'],
        'genre_SetRougeL': genre_rougeL,
        # genre extra
        'genre_extra_pred': pred_data_song.get('genre_extra', []),
        'genre_extra_anno': anno_data_song['genre_extra'],
        'genre_extra_SetRougeL': genre_extra_rougeL,
        # mood
        'mood_pred': pred_data_song.get('mood', []),
        'mood_anno': anno_data_song['mood'],
        'mood_SetRougeL': mood_rougeL,
        # language
        'language_pred': pred_data_song.get('language', []),
        'language_anno': anno_data_song['language'],
        'language_F1': language_f1,
        # bpm
        'bpm_pred': bpm_pred,
        'bpm_anno': bpm_anno,
        'bpm_Abs': tempo_abs,
        # tempo
        'tempo_pred': tempo_pred,
        'tempo_anno': tempo_anno,
        'tempo_F1': tempo_f1,
        # gender
        'gender_pred': pred_data_song.get('gender', []),
        'gender_anno': anno_data_song['gender'],
        'gender_F1': gender_f1,
        # scene
        'scene_pred': pred_data_song.get('scene', []),
        'scene_anno': anno_data_song['scene'],
        'scene_SetRougeL': scene_rougeL,
        # scene phrase
        'scene_phrase_pred': pred_data_song.get('scene_phrase', []),
        'scene_phrase_anno': anno_data_song['scene_phrase'],
        'scene_phrase_SetRougeL': scene_phrase_rougeL,
        # timbre
        'timbre_pred': pred_data_song.get('timbre', []),
        'timbre_anno': anno_data_song['timbre'],
        'timbre_F1': timbre_f1,
        'timbre_SetRougeL': timbre_rougeL,
        # instrument
        'instrument_pred': pred_data_song.get('instrument', []),
        'instrument_anno': anno_data_song['instrument'],
        'instrument_F1': instrument_f1,
        'instrument_Precision': instrument_precision,
        'instrument_Recall': instrument_recall,
        'instrument_SetRougeL': instrument_rougeL,
        # era
        'era_pred': pred_data_song.get('era', []),
        'era_anno': anno_data_song['era'],
        'era_F1': era_f1,
        # melody
        'melody_pred': pred_data_song.get('melody', []),
        'melody_anno': anno_data_song['melody'],
        'melody_SetRougeL': melody_rougeL,
        # rhythm
        'rhythm_pred': pred_data_song.get('rhythm', []),
        'rhythm_anno': anno_data_song['rhythm'],
        'rhythm_SetRougeL': rhythm_rougeL,
        # key
        'key_pred': pred_data_song.get('key', []),
        'key_anno': anno_data_song['key'],
        'key_F1': key_f1,
        # imagery
        'imagery_pred': pred_data_song.get('imagery', []),
        'imagery_anno': anno_data_song['imagery'],
        'imagery_SetRougeL': imagery_rougeL,
        # synesthesia tags
        'synesthesia_tags_pred': pred_data_song.get('synesthesia_tags', []),
        'synesthesia_tags_anno': anno_data_song['synesthesia_tags'],
        'synesthesia_tags_SetRougeL': synesthesia_tags_rougeL,
        # vibe
        'vibe_pred': pred_data_song.get('vibe', []),
        'vibe_anno': anno_data_song['vibe'],
        'vibe_SetRougeL': vibe_rougeL,
        # audio features
        'audio_features_pred': pred_data_song.get('audio_features', []),
        'audio_features_anno': anno_data_song['audio_features'],
        'audio_features_F1': audio_features_f1,
        # additional
        'additional_pred': pred_data_song.get('additional', []),
        'additional_anno': anno_data_song['additional'],
        'additional_SetRougeL': additional_rougeL,
        # description
        'description_pred': pred_data_song.get('description', [''])[0],
        'description_anno': anno_data_song['description'][0],
        'description_RougeL': description_rougeL,
        'description_Len_pred': len(pred_data_song.get('description', [''])[0]),
        'description_Len_anno': len(anno_data_song['description'][0]),
        # description long
        'description_long_pred': pred_data_song.get('description_long', [''])[0],
        'description_long_anno': anno_data_song['description_long'][0],
        'description_long_RougeL': description_long_rougeL,
        'description_long_Len_pred': len(pred_data_song.get('description_long', [''])[0]),
        'description_long_Len_anno': len(anno_data_song['description_long'][0]),
    }
    return result


def run():
    ##### Load data #####
    testset_id = os.getenv("DATASET_ID", DATASET_ID)
    with open(f'./gt_dict_{testset_id}.json', 'r') as f:
        anno_data = json.load(f)
    with open('results_merged.json', 'r') as f:
        pred_data = json.load(f)

    # Main loop
    rows = []
    error_cnt = 0
    for uttid, pred_data in pred_data.items():
        print('----------')
        print('uttid:', uttid)
        pred_text = pred_data['predict_results']
        pred_text_len = len(pred_text)
        url = ""

        try:
            anno_data_song = anno_data[uttid]
            url = anno_data_song['url']
        except:
            print('!!! key error', uttid)
            error_cnt += 1
            continue

        # parse and evaluate
        anno_data_song.pop('url', None)

        import ast
        try:
            pred_data_song = ast.literal_eval(pred_text)
        except Exception as e:
            print("Error parsing string:", e)
            data_dict = None

        with open("ttmpt.json", "w") as f:
            json.dump(pred_data_song, f)

       
        pred_data_song = _parse_gemini_v2(pred_data_song)
        result = evaluate(pred_data_song, anno_data_song)

        # update row
        row = {"uttid": uttid, "url": url, 'str_len': pred_text_len}
        row.update({k: (",".join(v) if isinstance(v, list) else v) for k, v in result.items()})
        rows.append(row)
        # import pdb; pdb.set_trace()

    ##### Export CSVs #####
    # main csv
    csv_file = "case_study.csv"
    header = rows[0].keys() if rows else []
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(csv_file, index=False, encoding='utf-8')

    num_rows = len(rows)
    
    # QA
    numeric_df = df.select_dtypes(include="number")
    text_df = df.select_dtypes(exclude="number")
    means = numeric_df.mean().round(3)
    means_df = means.to_frame(name="mean").T 
    means_df["num_rows"] = len(df)

    # Optional: save to CSV
    text_df.to_csv("text_only.csv", index=False)
    means_df.to_csv("numeric_means.csv")

    print('\n\n\n========================')
    print(f" [o] Saved result to {csv_file}. {num_rows} rows.")


if __name__ == '__main__':
    merge_result_data('results')
    run()