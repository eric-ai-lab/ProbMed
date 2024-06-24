import json
from collections import defaultdict
import numpy as np

def parse_response(models):
    '''
    parse response data from aggregated ans file into modality-organ types
    '''
    all_model_data = {}
    for model_name in models:
        response_data = {}
        with open(f"response_file/{model_name}.json", 'r') as f:
            response = json.load(f)
        for data in response:
            if data["image_type"] not in response_data:
                response_data[data["image_type"]] = [data]
            else: response_data[data["image_type"]].append(data)
        all_model_data[model_name] = response_data
    
    return all_model_data


def get_score_binary(response, ans):
    '''
    get binary score used for main results and ablation accuracy
    '''
    response = response.strip()
    if ans == 'yes':
        if 'Yes' in response or response.lower() == 'yes' or response.lower() == 'yes.': return 1
        else: return 0
    else: 
        if 'No' in response or response.lower() == 'no' or response.lower() == 'no.': return 1
        else: return 0

def get_score_dict(response_data, get_score):
    '''
    get score dict according to probmed data setting for later geting float scores
    '''
    cur_img_id = response_data[0]['id']
    score = defaultdict(list)
    score['id'] = [cur_img_id]

    modality_score = []
    body_part_score = []
    entity_score = []
    grounding_score = []

    for data in response_data:
        if data['id'] != cur_img_id: # next image
            score['id'].append(data['id'])
            cur_img_id = data['id']
            if len(modality_score) != 2:
                modality_score = [] # one of questions unanswered
            score['modality'].append(modality_score)
            modality_score = []
            if len(body_part_score) != 2:
                body_part_score = []
            score['body_part'].append(body_part_score)
            body_part_score = []
            score['entity'].append(entity_score)
            entity_id = -1
            entity_score = []
            score['grounding'].append(grounding_score)
            grounding_id = -1
            grounding_score = []
        if "modality" in data['qa_type']:
            modality_score.append(get_score(data['response'], data['gt_ans']))
        elif "body_part" in data['qa_type']:
            body_part_score.append(get_score(data['response'], data['gt_ans']))
        elif data['qa_type'] == 'abnormality':
            score['abnormality'].append(get_score(data['response'], data['gt_ans']))
        elif "entity" in data['qa_type']:
            if data['qa_type'] == "entity_hallu": # abnormality 0
                entity_score = [get_score(data['response'], data['gt_ans'])]
            else:
                if "gt" in data['qa_type']:
                    entity_id = data['qa_type'].split('_')[-1]
                    entity_score_tuple = [get_score(data['response'], data['gt_ans'])]
                else: 
                    if data['qa_type'].split('_')[-1] != entity_id: # gt question is not answered
                        continue
                    entity_score_tuple.append(get_score(data['response'], data['gt_ans']))
                    assert len(entity_score_tuple) == 2
                    entity_score.append(entity_score_tuple)
        else:
            if "gt" in data['qa_type']:
                grounding_id = data['qa_type'].split('_')[-1]
                grounding_score_tuple = [get_score(data['response'], data['gt_ans'])]
            else: 
                if data['qa_type'].split('_')[-1] != grounding_id: # gt question is not answered
                    continue
                grounding_score_tuple.append(get_score(data['response'], data['gt_ans']))
                assert len(grounding_score_tuple) == 2
                grounding_score.append(grounding_score_tuple)
    score['modality'].append(modality_score)
    score['body_part'].append(body_part_score)
    score['entity'].append(entity_score)
    score['grounding'].append(grounding_score)
    return score

def get_score_float(score):
    output_score = {}

    tmp = [d for d in score['abnormality'] if not np.isnan(d)]
    output_score['abnormality'] = {
        'acc' : sum(tmp) / len(tmp)*100,
        'num' : len(score['abnormality']),
    }

    tmp = []
    count_nan, count_all_ones, count_first_one, count_empty = 0, 0, 0, 0
    for t in score['modality']:
        if not t:
            count_empty += 1
            continue
        if np.isnan(t).any():
            count_nan += 1
        if all(elem == 1 for elem in t):
            assert not np.isnan(t).any()
            count_all_ones += 1
        if t[0] == 1:
            count_first_one += 1
    assert count_nan == 0
    output_score['modality'] = {
        'acc' : count_all_ones / ((len(score['modality'])-count_nan-count_empty))*100,
        'acc w. hallu': count_first_one / ((len(score['modality'])-count_nan-count_empty))*100,
        'num' : len(score['modality']) - count_empty
    }

    tmp = []
    count_nan, count_all_ones, count_first_one, count_empty = 0, 0, 0, 0
    for t in score['body_part']:
        if not t:
            count_empty += 1
            continue
        if np.isnan(t).any():
            count_nan += 1
        if all(elem == 1 for elem in t):
            assert not np.isnan(t).any()
            count_all_ones += 1
        if t[0] == 1:
            count_first_one += 1
    assert count_nan == 0
    output_score['body_part'] = {
        'acc' : count_all_ones / ((len(score['body_part'])-count_nan-count_empty))*100,
        'acc w. hallu': count_first_one / ((len(score['body_part'])-count_nan-count_empty))*100,
        'num' : len(score['body_part']) - count_empty
    }

    count_nan = 0
    filtered_list = []
    for l in score['entity']:
        if not l:
            continue
        if isinstance(l[0], list):  # Check if the first item is a list
            if all(np.nan in x for x in l):
                count_nan += 1
                continue
            filtered_list.append([x for x in l if np.nan not in x]) # remove [np,nan, 1] from l [[np,nan, 1], [0, 1]]
        else:
            if np.isnan(l[0]):
                count_nan += 1  # remove single [np.nan]
            else:
                filtered_list.append(l)
    count_first_1 = 0
    count_all_1 = 0
    for l in filtered_list:
        assert isinstance(l, list)
        if all(x[0] == 1 for x in (l if isinstance(l[0], list) else [l])):
            count_first_1 += 1
        if all(all(y == 1 for y in x) for x in (l if isinstance(l[0], list) else [l])):
            count_all_1 += 1
    output_score['entity'] = {
        'acc' : count_all_1 / len(filtered_list)*100,
        'acc w. hallu' : count_first_1 / len(filtered_list)*100,
        'num' : len(score['entity'])
    }

    filtered_list = []
    count_nan = 0
    count_empty = 0
    for l in score['grounding']:
        if not l:  # skip empty lists
            count_empty += 1
            continue
        if all(np.nan in x for x in l):
            count_nan += 1 
            continue
        filtered_list.append([x for x in l if np.nan not in x])
    count_first_1 = 0
    count_all_1 = 0
    for l in filtered_list:
        if isinstance(l, list) and all(isinstance(x, list) for x in l):  # check for list of lists
            if all(x[0] == 1 for x in (l if isinstance(l[0], list) else [l])):
                count_first_1 += 1
            if all(all(y == 1 for y in x) for x in l):
                count_all_1 += 1
    output_score['grounding'] = {
        'acc' : count_all_1 / len(filtered_list)*100,
        'acc w. hallu' : count_first_1 / len(filtered_list)*100,
        'num' : len(score['grounding'])
    }

    return output_score

def get_scores_probmed(all_model_data):
    '''
    all_scores: score per modality_body_part: [KEY] acc, acc w.o. adv pair, num (Tables in Appendix)
    all_scores_aggr_question: aggregated score per question type: [KEY] acc, acc w.o. adv pair (Table 5 results)
    overall_scores_aggr_question: overall aggregated score per question type: [KEY] acc, acc w.o. adv pair (Table 5 last column)
    '''
    all_scores = {}
    all_scores_aggr_question = {}
    overall_scores_aggr_question = {}
    for model_name, model_response in all_model_data.items():
        for image_type, response in model_response.items():       
            score_dict = get_score_dict(response, get_score=get_score_binary)
            score_per_cat = get_score_float(score_dict)
            if model_name not in all_scores:
                all_scores[model_name] = {}
            all_scores[model_name][image_type] = score_per_cat
        aggregated = {}
        for modality, questions in all_scores[model_name].items():
            for question, metrics in questions.items():
                if question not in aggregated:
                    aggregated[question] = {
                        "acc": 0,
                        "num": 0
                    }
                    if "acc w. hallu" in metrics:
                        aggregated[question]["acc w. hallu"] = 0
                aggregated[question]["acc"] += metrics["acc"] * metrics["num"]
                aggregated[question]["num"] += metrics["num"]
                if "acc w. hallu" in metrics:
                    aggregated[question]["acc w. hallu"] += metrics["acc w. hallu"] * metrics["num"]
        for question, metrics in aggregated.items():
            if metrics["num"] > 0:
                metrics["acc"] /= metrics["num"]
                if "acc w. hallu" in metrics:
                    metrics["acc w. hallu"] /= metrics["num"]
        all_scores_aggr_question[model_name] = aggregated

    for model, question_score in all_scores_aggr_question.items():
        overall_scores_aggr_question[model] = {
            "acc": 0,
            "num": 0,
            "acc w.o. adv pair": 0,
            "num w.o. adv pair": 0
        }
        for question, metrics in question_score.items():
            overall_scores_aggr_question[model]["acc"] += metrics["acc"] * metrics["num"]
            overall_scores_aggr_question[model]["num"] += metrics["num"]
            if "acc w. hallu" in metrics:
                overall_scores_aggr_question[model]["acc w.o. adv pair"] += metrics["acc w. hallu"] * metrics["num"]
                overall_scores_aggr_question[model]["num w.o. adv pair"] += metrics["num"]
        if overall_scores_aggr_question[model]["num"] > 0:
            overall_scores_aggr_question[model]["acc"] /= overall_scores_aggr_question[model]["num"]
        if overall_scores_aggr_question[model]["num w.o. adv pair"] > 0:
            overall_scores_aggr_question[model]["acc w.o. adv pair"] /= overall_scores_aggr_question[model]["num w.o. adv pair"]

    return all_scores, all_scores_aggr_question, overall_scores_aggr_question

def get_model_score_vqa_rad_ablation(ans_file_name):
    response_data = []
    if "jsonl" in ans_file_name:
        with open(ans_file_name, 'r') as f:
            for line in f:
                response_data.append(json.loads(line)) 
    else:
        with open(ans_file_name, 'r') as f:
            response_data = json.load(f)
    score = []
    for i, data in enumerate(response_data):
        if i % 2 == 0:
            assert data['gt_ans'] == 'yes'
            tmp = []
            tmp.append(get_score_binary(data['response'], 'yes'))
        else:
            assert data['gt_ans'] == 'no'
            tmp.append(get_score_binary(data['response'], 'no'))
            score.append(tmp)
    score_wo_adv = []
    score_w_adv = []
    for tmp in score:
        if tmp[0] == 1:
            score_wo_adv.append(1)
            if tmp[1] == 1:
                score_w_adv.append(1)
            else: score_w_adv.append(0)
        else:
            score_w_adv.append(0)
            score_wo_adv.append(0)
    assert len(score_w_adv) == len(score_wo_adv)
    return sum(score_w_adv)/len(score_w_adv), sum(score_wo_adv)/len(score_wo_adv)

def main():
    models = ["chexagent", "gemini", "gpt4v", "llava_v1.6", "llava_v1", "llavamed", "minigptv2", "gpt4o", "med-flamingo", "biomedgpt"]
    all_model_data = parse_response(models)
    all_scores, all_scores_aggr_question, overall_scores_aggr_question = get_scores_probmed(all_model_data)

    # # uncomment the block to print fine-grained accuracy
    # print('=== Printing accuracy in Appendix Tables ===')
    # for model, v in all_scores.items():
    #     for image_type, s in v.items():
    #         print(model, image_type)
    #         print(s)
    # print('=' * 30)

    print('=== Printing accuracy aggregated over modality-organ ===')
    for model, v in all_scores_aggr_question.items():
        print(model, v)
    print('=' * 30)

    print('=== Printing overall accuracy further aggregated over question types and difference w.&w.o. adv. pairs ===')
    for model, overall_score in overall_scores_aggr_question.items():
        print(f"{model} acc. w.o. adv. pair: {overall_score['acc w.o. adv pair']}, acc. w. adv. pair: {overall_score['acc w.o. adv pair']}, acc. diff: {overall_score['acc w.o. adv pair']}")
    print('=' * 30)

    print('=== Printing accuracy on ablation set and difference w.&w.o. adv. pairs ===')
    model_names = ["llava_v1.jsonl", "llava_v1.6.jsonl", "llavamed.jsonl", "minigptv2.jsonl", "chexagent.jsonl", "gpt4v.json", "gemini.jsonl", "gpt4o.json", "med-flamingo.jsonl", "biomedgpt.json"]
    summ = []
    for model in model_names:
        score = get_model_score_vqa_rad_ablation(f'ablation/{model}')
        print(f"{model} acc. w.o. adv. pair: {score[1]*100}, acc. w. adv. pair: {score[0]*100}, acc. diff: {score[1]*100 - score[0]*100}")
        summ.append(score[1]*100 - score[0]*100)
    print(f"average drop: {sum(summ)/len(summ)}")
    print('=' * 30)


if __name__ == "__main__":
    main()
