import re
import json
from collections import defaultdict
import json
import numpy as np
import pandas as pd

def extract_enetity(response):
    ent_dict = {}
    pattern = r"<([^>]+)> : <([^>]+)>"
    matches = re.findall(pattern, response)
    for match in matches:
        ent_dict[match[0].strip()] = match[1].strip()
        
    return ent_dict

def get_score(response, ans):
    response = response.strip()
    if ans == 'yes':
        if 'Yes' in response or response.lower() == 'yes': return 1
        elif 'No' in response or response.lower() == 'no': return 0
        else: return np.nan
    else: 
        if 'Yes' in response or response.lower() == 'yes': return 0
        elif 'No' in response or response.lower() == 'no': return 1
        else: return np.nan

# calculate score from response file for ProbMed
def get_model_score(ans_file_name, question_file_name):
    response_data = []

    if "jsonl" in ans_file_name:
        with open(ans_file_name, 'r') as f:
            for line in f:
                response_data.append(json.loads(line)) 
    else:
        with open(ans_file_name, 'r') as f:
            response_data = json.load(f)
    print(f"Evaluate {ans_file_name}...")   
            
    with open(question_file_name, 'r') as f:
        questions = json.load(f)
    
    score_dict = get_score_dict(response_data)
    score = get_score_float(score_dict)
    
    return score

def get_score_dict(response_data):
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
            score['modality'].append(modality_score)
            modality_score = []
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
    lengths = [len(lst) for lst in score.values()]
    assert len(set(lengths)) == 1 # all scores have same length as #images, could be empty list
    return score

def get_score_float(score):
    output_score = {}

    tmp = [d for d in score['abnormality'] if not np.isnan(d)]
    output_score['abnormality'] = {
        'acc' : sum(tmp) / len(tmp)*100,
        'fail to ans' : 1 - len(tmp)/len(score['abnormality']),
        'num' : len(score['abnormality']),
    }

    tmp = []
    count_nan, count_all_ones, count_first_one = 0, 0, 0
    for t in score['modality']:
        if np.isnan(t).any():
            count_nan += 1
        if all(elem == 1 for elem in t):
            assert not np.isnan(t).any()
            count_all_ones += 1
        if t[0] == 1:
            count_first_one += 1
    output_score['modality'] = {
        'acc' : count_all_ones / ((len(score['modality'])-count_nan))*100,
        'acc w. hallu': count_first_one / ((len(score['modality'])-count_nan))*100,
        'fail to ans' : count_nan/len(score['modality']),
        'num' : len(score['modality'])
    }

    tmp = []
    count_nan, count_all_ones, count_first_one = 0, 0, 0
    for t in score['body_part']:
        if np.isnan(t).any():
            count_nan += 1
        if all(elem == 1 for elem in t):
            assert not np.isnan(t).any()
            count_all_ones += 1
        if t[0] == 1:
            count_first_one += 1
    output_score['body_part'] = {
        'acc' : count_all_ones / ((len(score['body_part'])-count_nan))*100,
        'acc w. hallu': count_first_one / ((len(score['body_part'])-count_nan))*100,
        'fail to ans' : count_nan/len(score['body_part']),
        'num' : len(score['body_part'])
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
                count_nan += 1  # Remove single [np.nan]
            else:
                filtered_list.append(l)
    # assert len(filtered_list) + count_nan == len(score['entity'])
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
        'fail to ans' : count_nan / len(score['entity']),
        'num' : len(score['entity'])
    }

    filtered_list = []
    count_nan = 0
    count_empty = 0
    for l in score['grounding']:
        if not l:  # Skip empty lists
            count_empty += 1
            continue
        if all(np.nan in x for x in l):
            count_nan += 1 
            continue
        filtered_list.append([x for x in l if np.nan not in x])
    # assert len(filtered_list) + count_nan + count_empty == len(score['grounding'])
    count_first_1 = 0
    count_all_1 = 0
    for l in filtered_list:
        if isinstance(l, list) and all(isinstance(x, list) for x in l):  # Check for list of lists
            if all(x[0] == 1 for x in (l if isinstance(l[0], list) else [l])):
                count_first_1 += 1
            if all(all(y == 1 for y in x) for x in l):
                count_all_1 += 1
    output_score['grounding'] = {
        'acc' : count_all_1 / len(filtered_list)*100,
        'acc w. hallu' : count_first_1 / len(filtered_list)*100,
        'fail to ans' : count_nan / len(score['grounding']),
        'num' : len(score['grounding'])
    }

    return output_score

# calculate score from response file for ablation on RAD-VQA
def get_model_score_ablation(ans_file_name):
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
            tmp.append(get_score(data['response'], 'yes'))
        else:
            assert data['gt_ans'] == 'no'
            tmp.append(get_score(data['response'], 'no'))
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

# calculate accuracy from response file for procedural diagnosis analysis
def get_conditional_accuracy(score):
    output = {}
    id_general = [i for i, _ in enumerate(score['id']) if score['modality'][i] == 1 and score['body_part'][i] == 1]
    id_abnormality = [i for i in id_general if score['abnormality'][i] == 1]
    output["Abnormality | General"] = {}
    output["Abnormality | General"]['acc'] = len(id_abnormality)/len(id_general)*100
    output["Abnormality | General"]['num'] = len(id_general)
    id_wrong_abnormality  = [i for i in id_general if i not in id_abnormality]
    if len(id_wrong_abnormality) > 0:
        count_wrong, count_reject = 0, 0
        for i in id_wrong_abnormality:
            if score['abnormality_orig'][i] == 0: count_wrong += 1
            else: count_reject += 1
        output["Abnormality | General"]['wrong ans'] = count_wrong/len(id_wrong_abnormality)*100
        output["Abnormality | General"]['reject to ans'] = count_reject/len(id_wrong_abnormality)*100
    ##
    id_condition_non_empty = [i for i in id_abnormality if score['entity_orig'][i]] # collect non-empty
    id_condition = [i for i in id_condition_non_empty if score['entity'][i] == 1]
    output["Condition | General & Abnormality"] = {}
    output["Condition | General & Abnormality"]['acc'] = len(id_condition)/len(id_condition_non_empty)*100
    output["Condition | General & Abnormality"]['num'] = 0
    id_wrong_condition  = [i for i in id_condition_non_empty if i not in id_condition]
    if len(id_wrong_condition) > 0:
        count_deny_exist, count_fail_deny_hall, count_reject = 0, 0, 0
        for i in id_wrong_condition:
            if not score['grounding_orig'][i]: continue
            output["Condition | General & Abnormality"]['num'] += 1
            if not isinstance(score['entity_orig'][i][0], list):
                if score['entity_orig'][i][0] == 0: count_fail_deny_hall += 1
                else: count_reject += 1
            else:
                first_s = [s[0] for s in score['entity_orig'][i]]
                second_s = [s[1] for s in score['entity_orig'][i]]
                if np.isnan(first_s).any() or np.isnan(second_s).any(): count_reject += 1
                first_s = [s for s in first_s if not np.isnan(s)]
                second_s = [s for s in second_s if not np.isnan(s)]
                if not all(s == 1 for s in first_s): count_deny_exist += 1
                if not all(s == 1 for s in second_s): count_fail_deny_hall += 1
        output["Condition | General & Abnormality"]['deny gt'] = count_deny_exist/(count_reject+count_deny_exist+count_fail_deny_hall)*100
        output["Condition | General & Abnormality"]['fail to deny hallu'] = count_fail_deny_hall/(count_reject+count_deny_exist+count_fail_deny_hall)*100
        output["Condition | General & Abnormality"]['reject to ans'] = count_reject/(count_reject+count_deny_exist+count_fail_deny_hall)*100
    ##
    id_position_non_empty = [i for i in id_condition if score['grounding_orig'][i]] # collect non-empty
    id_position = [i for i in id_position_non_empty if score['grounding'][i] == 1]
    output["Position | General & Abnormality & Condition"] = {}
    if len(id_position_non_empty) > 0:
        output["Position | General & Abnormality & Condition"]['acc'] = len(id_position)/len(id_position_non_empty)*100
    else:
        output["Position | General & Abnormality & Condition"]['acc'] = "n/a"
    output["Position | General & Abnormality & Condition"]['num'] = 0
    id_wrong_position  = [i for i in id_position_non_empty if i not in id_position]
    if len(id_wrong_position) > 0:
        count_deny_exist, count_fail_deny_hall, count_reject = 0, 0, 0
        for i in id_wrong_position:
            if not score['grounding_orig'][i]: continue
            if not isinstance(score['grounding_orig'][i][0], list):
                assert False
            else:
                output["Position | General & Abnormality & Condition"]['num'] += 1
                first_s = [s[0] for s in score['grounding_orig'][i]]
                second_s = [s[1] for s in score['grounding_orig'][i]]
                if np.isnan(first_s).any() or np.isnan(second_s).any(): count_reject += 1
                first_s = [s for s in first_s if not np.isnan(s)]
                second_s = [s for s in second_s if not np.isnan(s)]
                if not all(s == 1 for s in first_s): count_deny_exist += 1
                if not all(s == 1 for s in second_s): count_fail_deny_hall += 1
        output["Position | General & Abnormality & Condition"]['deny gt'] = count_deny_exist/(count_reject+count_deny_exist+count_fail_deny_hall)*100
        output["Position | General & Abnormality & Condition"]['fail to deny hallu'] = count_fail_deny_hall/(count_reject+count_deny_exist+count_fail_deny_hall)*100
        output["Position | General & Abnormality & Condition"]['reject to ans'] = count_reject/(count_reject+count_deny_exist+count_fail_deny_hall)*100
    conditional_score = output
    return conditional_score

# calculate score from response file per image level
def get_score_float_img(score):
    output_score = {}
    output_score['id'] = score['id']
    output_score['modality'] = []   
    for t in score['modality']:
        if all(elem == 1 for elem in t): output_score['modality'].append(1)
        else: output_score['modality'].append(0)

    output_score['body_part'] = []
    for t in score['body_part']:
        if all(elem == 1 for elem in t): output_score['body_part'].append(1)
        else: output_score['body_part'].append(0)

    output_score['abnormality'] = []
    output_score['abnormality_orig'] = []
    for t in score['abnormality']:
        output_score['abnormality_orig'].append(t)
        if t == 1: output_score['abnormality'].append(1)
        else: output_score['abnormality'].append(0)

    output_score['entity'] = []
    output_score['entity_orig'] = []
    for l in score['entity']:
        output_score['entity_orig'].append(l)
        if not l: # empty entity for this image
            output_score['entity'].append(l)
        else: # [[1,1], [1,0], [nan, 1]] abnormal image; [1] normal image
            if all(all(y == 1 for y in x) for x in (l if isinstance(l[0], list) else [l])):
                output_score['entity'].append(1)
            else:
                output_score['entity'].append(0)
    

    output_score['grounding'] = []
    output_score['grounding_orig'] = []
    for l in score['grounding']:
        output_score['grounding_orig'].append(l)
        if len(l) == 0: # empty entity for this image
            output_score['grounding'].append(l)
        else: # [[1,1], [1,0], [nan, 1]] abnormal image; [1] normal image
            if all(all(y == 1 for y in x) for x in (l if isinstance(l[0], list) else [l])):
                output_score['grounding'].append(1)
            else:
                output_score['grounding'].append(0)  

    return output_score