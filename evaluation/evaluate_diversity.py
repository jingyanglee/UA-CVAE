# -*- coding: utf-8 -*-
# @Time        : 2020/5/27 15:20
# @Author      : ssxy00
# @File        : evaluate_diversity.py
# @Description : this file is modified from https://github.com/rekriz11/DeDiv/blob/dfafd46f57b3b4b6184bee30a7190e75f34ae81a/analyze_diversity.py
import json
import argparse
import jsonlines
import collections
import numpy as np
from tqdm import tqdm
from bert.load_bert import bert_model
import nltk
nltk.download('wordnet')
from lexicalrichness import LexicalRichness
from rouge import Rouge
bert = bert_model()
rouge = Rouge()
def eval_bleu(candidates, ref):
    split = []
    score = []
    for cand in candidates:
        split.append(cand.split())
    ref = ref.split()
    for s in split:
        score.append(nltk.translate.bleu_score.sentence_bleu(ref, s, weights=(0, 0, 0, 1)))
    ave = sum(score) / len(score)
    return ave

def eval_meteor(candidates, ref):
    score = []
    for cand in candidates:
        score.append(nltk.translate.meteor_score.meteor_score(ref, cand))
    ave = sum(score) / len(score)
    return ave

def eval_rouge_l_f(candidates, ref):
    score = []
    for cand in candidates:
        try:
            score.append(rouge.get_scores(cand, ref)[0]['rouge-l']['f'])
        except ValueError:
            score.append(0.)
    ave = sum(score) / len(score)
    return ave

def eval_rouge_l_r(candidates, ref):
    score = []
    for cand in candidates:
        try:
            score.append(rouge.get_scores(cand, ref)[0]['rouge-l']['r'])
        except ValueError:
            score.append(0.)
    ave = sum(score) / len(score)
    return ave

def eval_rouge_l_p(candidates, ref):
    score = []
    for cand in candidates:
        try:
            score.append(rouge.get_scores(cand, ref)[0]['rouge-l']['p'])
        except ValueError:
            score.append(0.)
    ave = sum(score) / len(score)
    return ave

def eval_distinct_k(candidates, k):
    """The total number of k-grams divided by the total number of tokens
         over all the candidates.
      """
    kgrams = set()
    total = 0
    if isinstance(candidates, str):
        if len(candidates) >= k:
            for i in range(0, len(candidates) - k + 1):
                kgrams.add(tuple(candidates[i:i + k]))
            total += len(candidates)
    else:
        for cand in candidates:
            if len(cand) < k:
                continue
            for i in range(0, len(cand) - k + 1):
                kgrams.add(tuple(cand[i:i + k]))
            total += len(cand)
    if total == 0:
        return 0
    else:
        return len(kgrams) / total

def eval_distinct_k_v2(candidates, k):
    """The total number of k-grams divided by the total number of tokens
         over all the candidates.
      """
    kgrams = set()
    total = 0
    if isinstance(candidates, str):
        if len(candidates) >= k:
            for i in range(0, len(candidates) - k + 1):
                kgrams.add(tuple(candidates[i:i + k]))
            total = len(kgrams)/len(candidates)
    else:
        final = []
        for cand in candidates:
            kgrams = set()
            if len(cand) < k:
                continue
            for i in range(0, len(cand) - k + 1):
                kgrams.add(tuple(cand[i:i + k]))
            final.append(len(kgrams)/len(cand))
        try:
            total = sum(final) / len(final)
        except ValueError and ZeroDivisionError:
            print(final)
    if total == 0:
        return 0
    else:
        return total

def eval_entropy_k(candidates, k):
    """Entropy method which takes into account word frequency."""
    kgram_counter = collections.Counter()
    for cand in candidates:
        for i in range(0, len(cand) - k + 1):
            kgram_counter.update([tuple(cand[i:i + k])])

    counts = kgram_counter.values()
    s = sum(counts)
    if s == 0:
        # all of the candidates are shorter than k
        return 0
    return (-1.0 / s) * sum(f * np.log(f / s) for f in counts)

def eval_len(candidates):
    lengths = []
    for cand in candidates:
        lengths.append(len(cand.split(" ")))
    ave = sum(lengths) / len(lengths)
    return ave

def c_score_emo(candidates, context, prev_context, prev_res, new):
    score_all = []
    score_last = []
    if new == True:
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context))], [context]))
            score_last.append(bert.predict_label([cand for _ in range(len(context))], [context]))
        #print([context])
    else:
        context_list =[]
        context_list.append(prev_context)
        context_list.append(prev_res)
        con = context.replace(prev_context, '')
        con = con.replace(prev_res, '')
        context_list.append(con)
        #print(context_list)
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context))], context))
            score_last.append(bert.predict_label([cand for _ in range(len(context_list))], [context_list[-1]]))
        #print(context_list)
    ave_all = sum(score_all) / len(score_all)
    ave_last = sum(score_last) / len(score_last)

    #print(ave)
    return ave_all, ave_last

def c_score(candidates, context, golden, context_list, new):
    score_all = []
    score_last = []
    if new == True:
        context_list = []
        context_list.append(context)
        #print(context_list, len(context_list))
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context_list))], context_list))
            score_last.append(bert.predict_label([cand for _ in range(len(context_list))], context_list))
        context_list.append(golden)
    else:
        for con in context_list:
            context = context.replace(con, '')
        context_list.append(context)

        #print(context_list, len(context_list))
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context_list))], context_list))
            score_last.append(bert.predict_label([cand for _ in range(len(context_list))], [context_list[-1]]))
        context_list.append(golden)
    ave_all = sum(score_all) / len(score_all)
    ave_last = sum(score_last) / len(score_last)
    return ave_all, ave_last, context_list

def c_score_v2(candidates, context, golden, context_list, new):
    score_all = []
    score_last = []
    if new == True:
        context_list = []
        context_list.append(context)
        #print(context_list, len(context_list))
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context_list))], context_list))
        context_list.append(golden)
    else:
        for con in context_list:
            context = context.replace(con, '')
        context_list.append(context)
        if len(context_list) > 4:
            context_list = context_list[-3:]
        #print(context_list, len(context_list))
        for cand in candidates:
            score_all.append(bert.predict_label([cand for _ in range(len(context_list))], context_list))
        context_list.append(golden)
    ave_all = sum(score_all) / len(score_all)
    return ave_all,context_list

def main(args):
    print(args.eval_file)
    ave_length = 0.
    average_distinct_1 = 0.
    average_distinct_2 = 0.
    average_distinct_3 = 0.
    average_entropy_4 = 0.
    cscore_all = 0.
    cscore_last = 0.
    average_bleu =0.
    average_rouge_l_f = 0.
    average_rouge_l_p = 0.
    average_rouge_l_r = 0.
    average_meteor = 0.
    context_list = ['placeholder']
    lengths = []
    with jsonlines.open(args.eval_file) as reader:
        for idx, row in enumerate(tqdm(reader)):
            lengths.append(eval_len(row["predict_responses"]))
            #ave_length = (ave_length * idx + eval_len(row["predict_responses"])) / (idx + 1)
            average_distinct_1 = (average_distinct_1 * idx + eval_distinct_k_v2(row["predict_responses"], 1)) / (idx + 1)
            average_distinct_2 = (average_distinct_2 * idx + eval_distinct_k_v2(row["predict_responses"], 2)) / (idx + 1)
            average_distinct_3 = (average_distinct_3 * idx + eval_distinct_k_v2(row["predict_responses"], 3)) / (idx + 1)
            average_bleu = (average_bleu * idx + eval_bleu(row["predict_responses"], row['golden_response'])) / (idx + 1)
            average_entropy_4 = (average_entropy_4 * idx + eval_entropy_k(row["predict_responses"], 4)) / (idx + 1)
            average_rouge_l_f = (average_rouge_l_f * idx + eval_rouge_l_f(row["predict_responses"], row['golden_response'])) / (idx + 1)
            average_rouge_l_p = (average_rouge_l_p * idx + eval_rouge_l_p(row["predict_responses"], row['golden_response'])) / (idx + 1)
            average_rouge_l_r = (average_rouge_l_r * idx + eval_rouge_l_r(row["predict_responses"], row['golden_response'])) / (idx + 1)
            average_meteor = (average_meteor * idx + eval_meteor(row["predict_responses"], row['golden_response'])) / (idx + 1)
            #convAI2 eval
            if context_list[-1] in row["context"]:
                new = False
                c_all, context_list = c_score_v2(row["predict_responses"], row["context"], row["golden_response"], context_list, new)
                cscore_all = (cscore_all * idx + c_all) / (idx + 1)
            else:
                new = True
                c_all, context_list = c_score_v2(row["predict_responses"], row["context"], row["golden_response"],context_list, new)
                cscore_all = (cscore_all * idx + c_all) / (idx + 1)
            print(f"UE_score(All): {cscore_all}")

    lengths = list(filter(lambda num: num != 0, lengths))
    print(f"Average Length: {sum(lengths) / len(lengths)}")
    print(f"distinct 1: {average_distinct_1}")
    print(f"distinct 2: {average_distinct_2}")
    print(f"distinct 3: {average_distinct_3}")
    print(f"entropy 4: {average_entropy_4}")
    print(f"BLEU: {average_bleu}")
    print(f"Rouge_l F1: {average_rouge_l_f}")
    print(f"Rouge_l Precision: {average_rouge_l_p}")
    print(f"Rouge_l Recall: {average_rouge_l_r}")
    print(f"METEOR: {average_meteor}")
    print(f"UE_score(All): {cscore_all}")
    print(f"UE_score(Last): {cscore_last}")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="./results.jsonl")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
