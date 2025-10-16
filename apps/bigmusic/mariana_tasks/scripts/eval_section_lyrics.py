import json
import os
import glob
import csv
import argparse
from typing import Dict, List, Any, Tuple
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

# ######################################################################################
# --- 核心评估逻辑 (代码1的精确统计 + 代码2的混合分词) ---
# ######################################################################################

@dataclass
class EditOps:
    """一个用于存储编辑操作计数的数据类 (来自代码1)。"""
    subs: int = 0
    ins: int = 0
    dels: int = 0

    def edits(self) -> int:
        """返回总编辑数。"""
        return self.subs + self.ins + self.dels

    def clone(self):
        """返回此对象的副本。"""
        return EditOps(subs=self.subs, ins=self.ins, dels=self.dels)

def edit_distance(seq1: List[str], seq2: List[str]) -> EditOps:
    """
    基于Wagner-Fischer算法，计算两个序列间的详细编辑距离 (S/I/D)。
    (来自代码1，用于精确统计)
    """
    len_sent2 = len(seq2)
    dold = [EditOps(ins=i) for i in range(len_sent2 + 1)]
    dnew = [EditOps() for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = EditOps(dels=i)
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1].clone()
            else:
                sub_cost = dold[j - 1].edits()
                ins_cost = dnew[j - 1].edits()
                del_cost = dold[j].edits()
                
                if sub_cost <= ins_cost and sub_cost <= del_cost:  # 替换
                    dnew[j] = dold[j - 1].clone()
                    dnew[j].subs += 1
                elif ins_cost < sub_cost and ins_cost < del_cost:  # 插入
                    dnew[j] = dnew[j - 1].clone()
                    dnew[j].ins += 1
                else:  # 删除
                    dnew[j] = dold[j].clone()
                    dnew[j].dels += 1
        dold, dnew = dnew, dold
    return dold[-1]

def tokenize_mixed(text: str) -> List[str]:
    """
    混合分词器 (来自代码2)：
    - 中日韩 (CJK) 字符按单字切分。
    - 英文单词和数字按词切分。
    """
    if not isinstance(text, str): return []
    text = text.lower()
    # 模式更新以匹配 CJK 字符、单词（含撇号）和数字
    pattern = (
        r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|'  # CJK 字符
        r'[a-z0-9]+(?:\'[a-z0-9]+)*'                   # 英文单词/数字
    )
    return re.findall(pattern, text)

def compute_wer_stats(ref: str, hyp: str, tokenizer, min_ref_len: int = 1) -> Dict[str, Any] or None:
    """
    【新】计算详细的WER统计信息 (错误数, 参考长度等)。
    如果参考文本长度小于min_ref_len，则返回None，表示该样本不参与统计。
    """
    ref_tokens = tokenizer(ref)
    hyp_tokens = tokenizer(hyp)
    
    ref_len = len(ref_tokens)
    if ref_len < min_ref_len:
        return None

    edits = edit_distance(ref_tokens, hyp_tokens)
    total_errors = edits.edits()
    
    return {
        'wer': (total_errors / ref_len) if ref_len > 0 else 0.0,
        'ins_err': edits.ins,
        'sub_err': edits.subs,
        'del_err': edits.dels,
        'total_err': total_errors,
        'ref_len': ref_len,
        'hyp_len': len(hyp_tokens),
        'normalized_ref': ' '.join(ref_tokens),
        'normalized_hyp': ' '.join(hyp_tokens),
    }

# --- Section 相关函数 (来自代码2，略作调整) ---

def merge_consecurtive_sections(text: str) -> str:
    """
    合并连续的相同section tag, 并应用后处理规则。
    例如: [Verse]...[Chorus]...[Chorus]...[Verse]
    会变成: [Verse]...[Chorus]...[Verse]
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 使用re.split()将文本分割成标签和歌词片段，同时保留标签
    parts = re.split(r'(\[\w+\])', text)
    
    # 过滤掉可能由split产生的空字符串
    parts = [p for p in parts if p.strip()]

    # 标签转换
    seg_map = {
        "silence": "silence",
        "end": "end",
        "build": "verse",
        "fadein": "intro",
        "opening": "intro",
        "stutter": "chorus",
        "slow": "verse",
        "drumroll": "inst",
        "synth": "inst",
        "closing": "outro",
        "interlude": "inst",
        "mantra": "verse",
        "fade-out": "outro",
        "out": "outro",
        "guitar": "inst",
        "head": "inst",
        "loop": "inst",
    }

    substr_map = {
        "other": "other",
        "pre-chorus-and-chorus": "chorus",
        "verse-and-chorus": "chorus",
        "intro": "intro",
        "verse": "verse",
        "prechorus": "pre-chorus",
        "pre-chorus": "pre-chorus",
        "refrain": "chorus",
        "chorus": "chorus",
        "bridge": "bridge",
        "outro": "outro",
        "fadeout": "outro",
        "ending": "outro",
        "fadein": "intro",
        "inst": "inst",
        "solo": "inst",
        "break": "inst",
        "trans": "bridge",
        "gtr": "inst",
        "section": "verse",
        "riff": "inst",
        "rap": "verse",
        "coda": "outro",
        "interlude": "inst",
        "lead-in": "inst",
        "theme": "chorus",
        "development": "verse",
        "variation": "bridge",
        "impro": "inst",
        "guitar": "inst",
        "spoken": "inst",
        "trumpet": "inst",
        "applause": "inst",
        "voice": "inst",
        "stage": "inst",
        "banjo": "inst",
        "crowd": "inst",
        "pause": "inst",
        "tag": "inst",
        "hook": "chorus",   # @Yixiao Zhang, added on 2025.2.24
    }

    # 转换为全小写，并应用label替换规则
    for i, part in enumerate(parts):
        if re.match(r'^\[\w+\]$', part.strip()):
            part = part.strip("[]")
            part = part.lower()
            if part in seg_map:  # 先检查是否直接在seg_map中
                parts[i] = f"[{seg_map[part]}]"
            else:                # 检查是否在substr_map中
                for k in substr_map:
                    if k in part:
                        parts[i] = f"[{substr_map[k]}]"
                        break

    

    # 2. 遍历片段，通过比较相邻标签来决定是否保留
    output_parts = []
    last_tag = None
    
    for part in parts:
        # 检查当前片段是否是一个标签
        if re.match(r'^\[\w+\]$', part.strip()):
            # 如果是标签，且与上一个标签不同，则保留并更新last_tag
            if part != last_tag:
                output_parts.append(part)
                last_tag = part
            # 如果与上一个标签相同，则忽略
            else:
                continue
        else:
            # 如果是歌词或非标签文本，则直接保留
            output_parts.append(part)

    # 规则 1: 移除除第一个之外的所有 [Intro] 标签
    intro_found = False
    new_parts = []
    for p in output_parts:
        # 只保留第一个找到的 [Intro]
        if p.strip() == "[intro]":
            if not intro_found:
                new_parts.append(p)
                intro_found = True
        else:
            new_parts.append(p)
    output_parts = new_parts
    
    # 规则 2: 移除除最后一个之外的所有 [Outro] 标签
    outro_found = False
    new_parts = []
    # 从后往前遍历，只保留最后一个找到的 [Outro]
    for p in reversed(output_parts):
        if p.strip() == "[outro]":
            if not outro_found:
                new_parts.append(p)
                outro_found = True
        else:
            new_parts.append(p)
    # 遍历结束后，需要将列表重新反转回来
    output_parts = list(reversed(new_parts))

    # 规则 3: 移除末尾的 [Silence] 或 [End]
    while len(output_parts) > 0 and output_parts[-1].strip() in ["[silence]", "[end]"]:
        output_parts.pop()

    # 3. 重新拼接成一个字符串
    return "".join(output_parts).strip()

def prepare_mixed_token_list(text: str) -> List[List[Any]]:
    pad_text = "pad " * 10 + text + " pad" * 10
    tokens = re.findall(r'\[.*?\]|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*', pad_text)
    
    sections, buf = [], []
    for tok in tokens:
        if re.fullmatch(r'\[(.*?)\]', tok):
            if buf: sections.append(["[Prev_Content]", buf])
            sections.append([tok, []])
            buf = []
        else:
            buf.append(tok)
    if buf: sections.append(["[End_Content]", buf])

    context_sections, current_content = [], []
    for sec_name, content in sections:
        if content:
            current_content = content
        else:
            context_sections.append([sec_name, current_content, []])
            current_content = []

    for i in range(len(context_sections) - 1):
        if context_sections[i+1][0] != "[Prev_Content]":
             context_sections[i][2] = context_sections[i+1][1]
    
    return [[sec, before, after] for sec, before, after in context_sections]

def _wer_for_context(ref: List[str], hyp: List[str]) -> float:
    r_len, h_len = len(ref), len(hyp)
    if r_len == 0: return float('inf')
    d = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1): d[i][0] = i
    for j in range(h_len + 1): d[0][j] = j
    
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            
    return d[r_len][h_len] / r_len

def contains_chinese(text: str) -> bool:
    """检测文本是否包含中文字符"""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def evaluate_boundary_metrics(prediction: str, truth: str, wer_th: float = 0.3, context: int = 10, context_min: int = 5) -> Dict:
    # ... (这部分边界检测逻辑保持不变，因为它依赖于样本级别的WER) ...
    pre_sec_tokens = prepare_mixed_token_list(prediction)
    tru_sec_tokens = prepare_mixed_token_list(truth)
    
    truth_filtered = [t for t in tru_sec_tokens if len(t[1][-context:] + t[2][:context]) > context_min]
    pred_filtered = [p for p in pre_sec_tokens if len(p[1][-context:] + p[2][:context]) > context_min]

    if not truth_filtered or not pred_filtered:
        return {'bound_precision': None, 'bound_recall': None, 'label_precision': None, 'label_recall': None}

    gt_contexts = [[t[0], t[1][-context:] + t[2][:context]] for t in truth_filtered]
    pr_contexts = [[p[0], p[1][-context:] + p[2][:context]] for p in pred_filtered]

    # Recall
    br, lr = 0, 0
    for gt_label, gt_ctx in gt_contexts:
        wers = [_wer_for_context(pr_ctx, gt_ctx) for _, pr_ctx in pr_contexts]
        if not wers or min(wers) >= wer_th: continue
        hit_idx = np.argmin(wers)
        br += 1
        if pr_contexts[hit_idx][0] == gt_label: lr += 1
        
    # Precision
    bp, lp = 0, 0
    for pr_label, pr_ctx in pr_contexts:
        wers = [_wer_for_context(pr_ctx, gt_ctx) for _, gt_ctx in gt_contexts]
        if not wers or min(wers) >= wer_th: continue
        hit_idx = np.argmin(wers)
        bp += 1
        if gt_contexts[hit_idx][0] == pr_label: lp += 1

    bound_precision= bp / len(pr_contexts) if pr_contexts else 0
    bound_recall= br / len(gt_contexts) if gt_contexts else 0
    bound_f1 = 2 * bound_precision * bound_recall / (bound_precision + bound_recall) if bound_precision and bound_recall and (bound_precision + bound_recall) > 0 else 0
    label_precision= lp / len(pr_contexts) if pr_contexts else 0
    label_recall= lr / len(gt_contexts) if gt_contexts else 0
    label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall) if label_precision and label_recall and (label_precision + label_recall) > 0 else 0
        
    return {
        'bound_precision': round(bound_precision, 3),
        'bound_recall': round(bound_recall, 3),
        'bound_f1': round(bound_f1, 3),
        'label_precision': round(label_precision, 3),
        'label_recall': round(label_recall, 3),
        'label_f1': round(label_f1, 3),
    }

SECTION_LABELS_FOR_CM = ['[intro]', '[verse]', '[pre-chorus]', '[chorus]', '[bridge]', '[inst]', '[outro]']
DELETION_TOKEN = '<del>'    # 表示一个标签被删除了 (在真值中有，在预测中没有)
INSERTION_TOKEN = '<ins>' # 表示一个标签被插入了 (在预测中有，在真值中没有)


def align_sequences(ref: List[str], hyp: List[str]) -> List[tuple[str, str]]:
    """
    使用Wagner-Fischer算法的回溯路径来对齐两个序列。
    返回一个对齐后的 (真值, 预测) 对的列表。
    """
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    ops = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)  # 0: sub/match, 1: ins, 2: del

    for i in range(len(ref) + 1):
        d[i, 0] = i
        ops[i, 0] = 2 # Deletion
    for j in range(len(hyp) + 1):
        d[0, j] = j
        ops[0, j] = 1 # Insertion
    
    ops[0,0] = 0

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            
            sub_cost = d[i - 1, j - 1] + cost
            ins_cost = d[i, j - 1] + 1
            del_cost = d[i - 1, j] + 1
            
            costs = [sub_cost, ins_cost, del_cost]
            min_cost = min(costs)
            op_idx = costs.index(min_cost)

            d[i, j] = min_cost
            ops[i, j] = op_idx

    # 回溯
    alignment = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:  # Match or Substitution
            alignment.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif op == 1: # Insertion
            alignment.append((INSERTION_TOKEN, hyp[j-1]))
            j -= 1
        else: # Deletion
            alignment.append((ref[i-1], DELETION_TOKEN))
            i -= 1
            
    return list(reversed(alignment))

def analyze_and_print_cm_report(summary: Dict):
    """
    接收包含混淆矩阵的summary字典，计算详细指标并打印分析报告。
    """
    cm_data = summary.get('section_confusion_matrix')
    if not cm_data:
        print("\n混淆矩阵数据未找到，跳过详细分析。")
        return

    # 1. 从字典重建 DataFrame
    try:
        cm_df = pd.DataFrame(cm_data['matrix'], index=cm_data['index'], columns=cm_data['labels'])
    except Exception as e:
        print(f"\n无法重建混淆矩阵进行分析: {e}")
        return

    print("\n" + "=" * 80)
    print("混 淆 矩 阵 深 度 分 析")
    print("=" * 80)

    # 2. 计算总体性能指标
    # 创建一个不含 <ins>/<del> 的核心矩阵用于计算替换数和正确数
    core_labels = [l for l in SECTION_LABELS_FOR_CM if l in cm_df.index and l in cm_df.columns]
    core_cm = cm_df.loc[core_labels, core_labels]
    
    total_correct = np.diag(core_cm.values).sum()
    substitutions = core_cm.values.sum() - total_correct
    deletions = cm_df[DELETION_TOKEN].sum() if DELETION_TOKEN in cm_df.columns else 0
    insertions = cm_df.loc[INSERTION_TOKEN].sum() if INSERTION_TOKEN in cm_df.index else 0
    
    total_errors = substitutions + deletions + insertions
    # 真值总标签数 = 正确的 + 被替换的 + 被删除的
    total_gt_labels = total_correct + substitutions + deletions
    
    print("--- 1. 总体性能摘要 ---")
    if total_gt_labels > 0:
        accuracy = total_correct / total_gt_labels
        print(f"总标签数 (Ground Truth): {total_gt_labels}")
        print(f"正确预测数:             {total_correct}")
        print(f"错误预测总数:             {total_errors}")
        print(f"总体准确率:             {accuracy:.2%}")
    else:
        print("无有效标签可供分析。")

    print("\n--- 2. Per-Class Performance Metrics ---")
    header = f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support (True Count)':<15}"
    print(header)
    print("-" * len(header))

    for label in core_labels:
        true_positives = cm_df.loc[label, label]
        
        # Predicted Positives = Column Sum
        predicted_positives = cm_df[label].sum()
        # Actual Positives = Row Sum
        actual_positives = cm_df.loc[label].sum()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{label:<15} {precision:<12.2%} {recall:<12.2%} {f1_score:<12.2%} {int(actual_positives):<15d}")


    if total_errors == 0:
        print("\n模型表现完美，无错误可供分析！")
        print("=" * 80)
        return

    # 3. 计算错误类型分解
    print("\n--- 3. 错误类型分解 ---")
    print(f"删除 (Deletions):      {deletions:4d} 个 ({deletions/total_errors:.1%})")
    print(f"插入 (Insertions):     {insertions:4d} 个 ({insertions/total_errors:.1%})")
    print(f"替换 (Substitutions):  {substitutions:4d} 个 ({substitutions/total_errors:.1%})")

    # 4. 生成详细错误列表
    error_list = []
    for true_label, row in cm_df.iterrows():
        for pred_label, count in row.items():
            if count > 0 and true_label != pred_label:
                error_list.append({
                    "true": true_label,
                    "pred": pred_label,
                    "count": int(count)
                })
    
    # 按数量降序排序
    sorted_errors = sorted(error_list, key=lambda x: x['count'], reverse=True)
    
    # 5. 打印Top错误分析表格
    print("\n--- 4. Top 10 详细错误分析 ---")
    header = f"{'排名':<4} {'错误类型 (真值 → 预测)':<30} {'数量':<6} {'占比':<8} {'累计占比':<10}"
    print(header)
    print("-" * len(header))
    
    cumulative_percentage = 0.0
    fifty_percent_mark_printed = False
    for i, error in enumerate(sorted_errors):
        rank = i + 1
        true = error['true']
        pred = error['pred']
        count = error['count']
        
        percentage = count / total_errors
        cumulative_percentage += percentage
        
        error_str = f"{true} → {pred}"
        print(f"{rank:<4} {error_str:<30} {count:<6} {percentage:<8.1%} {cumulative_percentage:<10.1%}")

        # 标记出累计错误超过50%的位置
        if cumulative_percentage > 0.5 and not fifty_percent_mark_printed:
            print("-" * len(header) + " <-- 超过50%的错误源于以上类别")
            fifty_percent_mark_printed = True

        if rank == 10:
            break
            
    print("=" * 80)

# ############################################################################
# --- 主应用框架 (来自代码1的结构 + 代码2的逻辑) ---
# ############################################################################

def main():
    """主函数，用于处理和分析ASR及Section结构的结果。"""
    args = parse_arguments()
    print("ASR & Section 结构分析器 (已更新WER统计方法)")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n步骤 1: 合并结果文件...")
    merged_data = merge_results_from_ranks(args.input_dir, args.pattern)
    if not merged_data:
        print("未找到可处理的数据。")
        return
    print(f"合并了 {len(merged_data)} 个独立样本。")

    print("\n步骤 2: 计算性能指标...")
    summary, detailed = calculate_performance_metrics(merged_data)

    print_summary_report(summary)

    analyze_and_print_cm_report(summary)

    print("\n步骤 3: 保存结果...")
    if not args.no_csv:
        save_detailed_to_csv(detailed, args.output_dir)
        save_summary_to_csv(summary, args.output_dir)
    if not args.no_json:
        save_to_json(summary, detailed, args.output_dir)

    print(f"\n分析完成！请检查 {args.output_dir} 目录下的输出文件。")

def calculate_performance_metrics(merged_data: List[Dict]) -> Tuple[Dict, List]:
    """【已修改】遍历所有样本，使用基于计数的聚合方法计算性能指标，并区分中英文子集。"""
    detailed_results = []
    
    # 初始化总数计数器 - 全部
    total_lyrics = {'err': 0, 'ref': 0, 'ins': 0, 'sub': 0, 'del': 0}
    total_sections = {'err': 0, 'ref': 0}
    
    # 初始化计数器 - 含中文
    chinese_lyrics = {'err': 0, 'ref': 0, 'ins': 0, 'sub': 0, 'del': 0}
    
    # 初始化计数器 - 不含中文（纯英文/数字）
    english_lyrics = {'err': 0, 'ref': 0, 'ins': 0, 'sub': 0, 'del': 0}
    
    # 边界检测指标列表，用于后续求平均
    boundary_metrics = {k: [] for k in ["bound_precision", "bound_recall", "bound_f1", "label_precision", "label_recall", "label_f1"]}
    
    # 初始化混淆矩阵
    # 行代表真值 (True Labels)，列代表预测值 (Predicted Labels)
    cm_labels = SECTION_LABELS_FOR_CM
    true_labels_axis = cm_labels + [INSERTION_TOKEN]
    pred_labels_axis = cm_labels + [DELETION_TOKEN]
    confusion_matrix = pd.DataFrame(0, index=true_labels_axis, columns=pred_labels_axis)


    # 样本计数
    num_valid_lyrics_samples = 0
    num_valid_section_samples = 0
    num_chinese_samples = 0
    num_english_samples = 0

    for item in tqdm(merged_data):
        truth_text = item.get('lyrics', '')
        pred_text = item.get('predict_results', '')

        # 全小写
        truth_text = truth_text.lower()
        pred_text = pred_text.lower()

        # 如果出现了[Lyrics: xxx] 的pattern，替换为xxx
        pred_text = re.sub(r'\[lyrics:\s*(.*?)\s*\]', r'\1', pred_text, flags=re.DOTALL)

        # 如果出现了[Verse 1/2/3] 等pattern，替换为[Verse]
        pred_text = re.sub(r'\[(\w+)\s+\d+(?:/\d+)?\]', r'[\1]', pred_text)

        processed_truth_csv = merge_consecurtive_sections(truth_text)
        processed_pred_csv = merge_consecurtive_sections(pred_text)
        
        processed_truth = merge_consecurtive_sections(truth_text).replace('\n', ' ').replace('\\n', ' ')
        processed_pred = merge_consecurtive_sections(pred_text).replace('\n', ' ').replace('\\n', ' ')
        
        # 1. 计算歌词WER统计
        pred_lyrics = re.sub(r'\[.*?\]\n?', '', processed_pred)
        true_lyrics = re.sub(r'\[.*?\]\n?', '', processed_truth)
        
        # 检测是否为中文样本
        language = item.get('language', None)
        if language is None:
            is_chinese = contains_chinese(true_lyrics)
        else:
            is_chinese = True if language == 'ZH' else False
        
        if language != 'None':  # for nonvocal music, language == 'None', which is string 'None'
            lyrics_stats = compute_wer_stats(true_lyrics, pred_lyrics, tokenizer=tokenize_mixed, min_ref_len=10)
        else:
            lyrics_stats = {}
        
        if lyrics_stats:
            num_valid_lyrics_samples += 1
            total_lyrics['err'] += lyrics_stats['total_err']
            total_lyrics['ref'] += lyrics_stats['ref_len']
            total_lyrics['ins'] += lyrics_stats['ins_err']
            total_lyrics['sub'] += lyrics_stats['sub_err']
            total_lyrics['del'] += lyrics_stats['del_err']
            
            if is_chinese:
                num_chinese_samples += 1
                chinese_lyrics['err'] += lyrics_stats['total_err']
                chinese_lyrics['ref'] += lyrics_stats['ref_len']
                chinese_lyrics['ins'] += lyrics_stats['ins_err']
                chinese_lyrics['sub'] += lyrics_stats['sub_err']
                chinese_lyrics['del'] += lyrics_stats['del_err']
            else:
                num_english_samples += 1
                english_lyrics['err'] += lyrics_stats['total_err']
                english_lyrics['ref'] += lyrics_stats['ref_len']
                english_lyrics['ins'] += lyrics_stats['ins_err']
                english_lyrics['sub'] += lyrics_stats['sub_err']
                english_lyrics['del'] += lyrics_stats['del_err']

        # 2. 计算Section结构WER统计
        pred_sections_tokens = re.findall(r'\[.*?\]', processed_pred)
        true_sections_tokens = re.findall(r'\[.*?\]', processed_truth)
        
        pred_sections_str = ' '.join(pred_sections_tokens)
        true_sections_str = ' '.join(true_sections_tokens)

        section_stats = compute_wer_stats(true_sections_str, pred_sections_str, tokenizer=str.split, min_ref_len=3)

        # print(f"pred_sections_str: {pred_sections_str}")
        # print(f"true_sections_str: {true_sections_str}")
        
        if section_stats:
            num_valid_section_samples +=1
            total_sections['err'] += section_stats['total_err']
            total_sections['ref'] += section_stats['ref_len']

            # 对齐并填充混淆矩阵
            aligned_pairs = align_sequences(true_sections_tokens, pred_sections_tokens)
            for true_label, pred_label in aligned_pairs:
                # 确保标签在我们的矩阵索引中，忽略其他标签
                true_key = true_label if true_label in cm_labels or true_label == INSERTION_TOKEN else None
                pred_key = pred_label if pred_label in cm_labels or pred_label == DELETION_TOKEN else None
                
                if true_key and pred_key:
                    confusion_matrix.loc[true_key, pred_key] += 1

        # 3. 计算边界指标
        b_metrics = evaluate_boundary_metrics(processed_pred, processed_truth)
        for key, value_list in boundary_metrics.items():
            if b_metrics.get(key) is not None:
                value_list.append(b_metrics[key])

        detailed_results.append({
            'uttid': item.get('uttid', 'unknown'),
            'lyrics_wer': round(lyrics_stats['wer'], 3) if lyrics_stats else None,
            'section_wer': round(section_stats['wer'], 3) if section_stats else None,
            'is_chinese': is_chinese if lyrics_stats else None,
            **b_metrics,
            'lyrics_err_counts': (lyrics_stats['total_err'], lyrics_stats['ref_len']) if lyrics_stats else (0,0),
            'original_ref': truth_text,
            'original_hyp': pred_text,
            'processed_ref': processed_truth_csv,
            'processed_hyp': processed_pred_csv,
        })
    
    # 计算最终的聚合指标
    summary = {
        'total_samples': len(merged_data),
        'num_valid_lyrics_samples': num_valid_lyrics_samples,
        'num_valid_section_samples': num_valid_section_samples,
        'display_chinese_wer': "",
        'display_english_wer': "",
        'display_section_wer': "",
        'display_section_boundary': "",
        'display_section_label': "",
        
        # 中文子集指标
        'num_chinese_samples': num_chinese_samples,
        'chinese_lyrics_wer': (chinese_lyrics['err'] / chinese_lyrics['ref']) if chinese_lyrics['ref'] > 0 else 0.0,
        'chinese_lyrics_ins_rate': (chinese_lyrics['ins'] / chinese_lyrics['ref']) if chinese_lyrics['ref'] > 0 else 0.0,
        'chinese_lyrics_sub_rate': (chinese_lyrics['sub'] / chinese_lyrics['ref']) if chinese_lyrics['ref'] > 0 else 0.0,
        'chinese_lyrics_del_rate': (chinese_lyrics['del'] / chinese_lyrics['ref']) if chinese_lyrics['ref'] > 0 else 0.0,
        'chinese_total_lyrics_errors': chinese_lyrics['err'],
        'chinese_total_lyrics_words': chinese_lyrics['ref'],
        
        # 英文子集指标
        'num_english_samples': num_english_samples,
        'english_lyrics_wer': (english_lyrics['err'] / english_lyrics['ref']) if english_lyrics['ref'] > 0 else 0.0,
        'english_lyrics_ins_rate': (english_lyrics['ins'] / english_lyrics['ref']) if english_lyrics['ref'] > 0 else 0.0,
        'english_lyrics_sub_rate': (english_lyrics['sub'] / english_lyrics['ref']) if english_lyrics['ref'] > 0 else 0.0,
        'english_lyrics_del_rate': (english_lyrics['del'] / english_lyrics['ref']) if english_lyrics['ref'] > 0 else 0.0,
        'english_total_lyrics_errors': english_lyrics['err'],
        'english_total_lyrics_words': english_lyrics['ref'],
        
        # 总体指标
        'overall_lyrics_wer': (total_lyrics['err'] / total_lyrics['ref']) if total_lyrics['ref'] > 0 else 0.0,
        'overall_lyrics_ins_rate': (total_lyrics['ins'] / total_lyrics['ref']) if total_lyrics['ref'] > 0 else 0.0,
        'overall_lyrics_sub_rate': (total_lyrics['sub'] / total_lyrics['ref']) if total_lyrics['ref'] > 0 else 0.0,
        'overall_lyrics_del_rate': (total_lyrics['del'] / total_lyrics['ref']) if total_lyrics['ref'] > 0 else 0.0,
        'total_lyrics_errors': total_lyrics['err'],
        'total_lyrics_words': total_lyrics['ref'],
        
        # Section WER (基于总数)
        'overall_section_wer': (total_sections['err'] / total_sections['ref']) if total_sections['ref'] > 0 else 0.0,
        'total_section_errors': total_sections['err'],
        'total_section_tags': total_sections['ref'],
        
        # 边界指标 (基于平均)
        'avg_bound_precision': np.mean(boundary_metrics['bound_precision']) if boundary_metrics['bound_precision'] else 0.0,
        'avg_bound_recall': np.mean(boundary_metrics['bound_recall']) if boundary_metrics['bound_recall'] else 0.0,
        'avg_bound_f1': np.mean(boundary_metrics['bound_f1']) if boundary_metrics['bound_f1'] else 0.0,
        'avg_label_precision': np.mean(boundary_metrics['label_precision']) if boundary_metrics['label_precision'] else 0.0,
        'avg_label_recall': np.mean(boundary_metrics['label_recall']) if boundary_metrics['label_recall'] else 0.0,
        'avg_label_f1': np.mean(boundary_metrics['label_f1']) if boundary_metrics['label_f1'] else 0.0,

        # 将混淆矩阵添加到 summary 中 (转换为字典格式以便JSON序列化)
        'section_confusion_matrix': {
            'labels': confusion_matrix.columns.tolist(),
            'matrix': confusion_matrix.values.tolist(),
            'index': confusion_matrix.index.tolist()
        }
    }
    summary['display_chinese_wer'] = f"{summary['chinese_lyrics_wer']:.1%} ({summary['chinese_lyrics_ins_rate']*100:.1f}, {summary['chinese_lyrics_sub_rate']*100:.1f}, {summary['chinese_lyrics_del_rate']*100:.1f})"
    summary['display_english_wer'] = f"{summary['english_lyrics_wer']:.1%} ({summary['english_lyrics_ins_rate']*100:.1f}, {summary['english_lyrics_sub_rate']*100:.1f}, {summary['english_lyrics_del_rate']*100:.1f})"
    summary['display_section_wer'] = f"{summary['overall_section_wer']:.1%}"
    summary['display_section_boundary'] = f"{summary['avg_bound_precision']:.3f}/{summary['avg_bound_recall']:.3f}/{summary['avg_bound_f1']:.3f}"
    summary['display_section_label'] = f"{summary['avg_label_precision']:.3f}/{summary['avg_label_recall']:.3f}/{summary['avg_label_f1']:.3f}"
    
    return summary, detailed_results

def print_summary_report(summary: Dict):
    """在控制台打印格式化的性能报告，包含中英文子集统计。"""
    print("\n" + "=" * 80)
    print("总 体 性 能 报 告")
    print("=" * 80)
    print(f"总样本数: {summary['total_samples']}")
    print(f"有效歌词评估样本数: {summary['num_valid_lyrics_samples']}")
    print(f"  - 中文样本数: {summary['num_chinese_samples']}")
    print(f"  - 英文样本数: {summary['num_english_samples']}")
    print(f"有效Section评估样本数: {summary['num_valid_section_samples']}")
    print("-" * 80)
    
    print("--- 歌词识别性能 (Lyrics Performance) ---")
    print(f"总参考词数 (中/英): {summary['total_lyrics_words']}")
    print(f"总错误数 (S+D+I): {summary['total_lyrics_errors']}")
    print(f"总体歌词 WER: {summary['overall_lyrics_wer']:.2%}")
    print(f"  - 替换率: {summary['overall_lyrics_sub_rate']:.2%}")
    print(f"  - 删除率: {summary['overall_lyrics_del_rate']:.2%}")
    print(f"  - 插入率: {summary['overall_lyrics_ins_rate']:.2%}")
    
    print("\n--- 中文歌词性能 (Chinese Lyrics) ---")
    print(f"中文参考词数: {summary['chinese_total_lyrics_words']}")
    print(f"中文错误数: {summary['chinese_total_lyrics_errors']}")
    print(f"中文歌词 WER: {summary['chinese_lyrics_wer']:.2%}")
    print(f"  - 替换率: {summary['chinese_lyrics_sub_rate']:.2%}")
    print(f"  - 删除率: {summary['chinese_lyrics_del_rate']:.2%}")
    print(f"  - 插入率: {summary['chinese_lyrics_ins_rate']:.2%}")
    
    print("\n--- 英文歌词性能 (English Lyrics) ---")
    print(f"英文参考词数: {summary['english_total_lyrics_words']}")
    print(f"英文错误数: {summary['english_total_lyrics_errors']}")
    print(f"英文歌词 WER: {summary['english_lyrics_wer']:.2%}")
    print(f"  - 替换率: {summary['english_lyrics_sub_rate']:.2%}")
    print(f"  - 删除率: {summary['english_lyrics_del_rate']:.2%}")
    print(f"  - 插入率: {summary['english_lyrics_ins_rate']:.2%}")
    
    print("-" * 80)
    print("--- Section 结构性能 (Structure Performance) ---")
    print(f"总参考Section数: {summary['total_section_tags']}")
    print(f"总Section错误数: {summary['total_section_errors']}")
    print(f"总体Section WER: {summary['overall_section_wer']:.2%}")
    print(f"标签P/R: {summary['avg_label_precision']:.2%}/{summary['avg_label_recall']:.2%}")
    print(f"边界P/R: {summary['avg_bound_precision']:.2%}/{summary['avg_bound_recall']:.2%}")
    print("=" * 80)

    # 打印混淆矩阵
    cm_data = summary.get('section_confusion_matrix')
    if cm_data:
        cm_df = pd.DataFrame(cm_data['matrix'], index=cm_data['index'], columns=cm_data['labels'])
        print("\n--- Section 混淆矩阵 (行: 真值, 列: 预测) ---")
        # 使用 to_string() 方法以获得更好的对齐效果
        print(cm_df.to_string())

# --- 文件合并与保存函数 (与原代码2基本一致) ---

def merge_results_from_ranks(results_dir: str, pattern: str) -> List[Dict]:
    merged_data = {}
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        print(f"警告: 未找到匹配模式的文件: {os.path.join(results_dir, pattern)}")
        return []

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
            
            # 尝试从文件名提取rank，如果没有则使用默认值
            rank_match = re.search(r'rank_(\d+)', os.path.basename(file_path))
            rank = int(rank_match.group(1)) if rank_match else 0
            
            # 如果数据是单个dict而不是list，转换为list
            if isinstance(data, dict):
                data = [data]
            
            for item in data:
                if (uttid := item.get('uttid')):
                    # 如果没有rank信息，直接添加或覆盖
                    if rank == 0 or uttid not in merged_data or rank < merged_data[uttid].get('rank', float('inf')):
                        item['rank'] = rank
                        merged_data[uttid] = item
        except Exception as e: 
            print(f"错误: 处理文件 {file_path} 时失败: {e}")
    return list(merged_data.values())

def save_detailed_to_csv(detailed_results: List[Dict], output_dir: str):
    csv_file = os.path.join(output_dir, "detailed_performance_results.csv")
    fieldnames = ['uttid', 'lyrics_wer', 'section_wer', 'is_chinese', 'bound_precision', 'bound_recall', 
                  'label_precision', 'label_recall', 'lyrics_err_counts', 'original_ref', 'original_hyp', 'processed_ref', 'processed_hyp']
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"详细结果已保存至: {csv_file}")
    except Exception as e: print(f"错误: 保存详细CSV时失败: {e}")

def save_summary_to_csv(summary: Dict, output_dir: str):
    csv_file = os.path.join(output_dir, "summary_performance_metrics.csv")
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in summary.items():
                val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                writer.writerow([key, val_str])
        print(f"汇总指标已保存至: {csv_file}")
    except Exception as e: print(f"错误: 保存汇总CSV时失败: {e}")

def save_to_json(summary: Dict, detailed_results: List[Dict], output_dir: str):
    output_file = os.path.join(output_dir, "analysis_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'performance_summary': summary, 'detailed_results': detailed_results}, 
                      f, indent=2, ensure_ascii=False)
        print(f"JSON结果已保存至: {output_file}")
    except Exception as e: print(f"错误: 保存JSON结果时失败: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASR & Section 结构分析器 - 使用混合分词WER和Section评估方法。')
    parser.add_argument('--input-dir', '-i', type=str, default='/mlx_devbox/users/yixiao.zhang/playground/results_new', help='输入JSON文件目录')
    parser.add_argument('--output-dir', '-o', type=str, default='./results_new/', help='输出报告目录')
    parser.add_argument('--pattern', '-p', type=str, default='result.json', help='结果文件匹配模式')
    parser.add_argument('--no-csv', action='store_true', help='不保存CSV报告')
    parser.add_argument('--no-json', action='store_true', help='不保存JSON报告')
    return parser.parse_args()

if __name__ == "__main__":
    main()