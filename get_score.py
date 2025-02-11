import json
import ast
# path = "summary_only_LLM_evaluate.json"
path = "data_evaluate.json"

def get_multiple(data):
    mkey = []
    for key in data.keys():
        line = data[key]
        parsing_result = line['parsing']

        for i in range(1, len(parsing_result)):
            parsing_line = parsing_result[i]
            reply = parsing_line[3]
            if(reply is None or reply['related_line_id'] is None):
                mkey.append(key)
                break
    return mkey
mkey = ["train_574", "train_767", "train_812", "train_1049", "train_1100", "train_1379", "train_1623", "train_2108", "train_2584", "train_3066", "train_3502", "train_4546", "train_4646", "train_6759", "train_7040", "train_7428", "train_7712", "train_8206", "train_8439", "train_8571", "train_9122", "train_9272", "train_9318", "train_9559", "train_10524", "train_10526", "train_10714", "train_11150", "train_11537", "train_11589", "train_12446"]


with open(path, "r") as f:
    data = json.load(f)


faithfulness_score_all = 0
completeness_score_all = 0
conciseness_score_all = 0
count = 0
map = {}

evaluated_ids = []

jump=[]

for line in data:
    try:
        if("faithfulness_score" not in line.keys()):
            continue
        if(line['id'] not in mkey):
            continue

        faithfulness_score=line["faithfulness_score"]
        completeness_score=line["completeness_score"]
        conciseness_score=line["conciseness_score"]

        faithfull = line['fact_checking_str']
        start_idx = faithfull.find('[')


        if start_idx != -1:
            end_idx = faithfull.find(']')
            faithfull = faithfull[start_idx:end_idx + 1]
            faithfull = faithfull.replace('\n', '')
            faithfull = ast.literal_eval(faithfull)


        faithfulness_score_all+=faithfulness_score
        completeness_score_all+=completeness_score
        conciseness_score_all+=conciseness_score
        count+=1
        evaluated_ids.append(line['id'])
    except:
        print(line['id'])


print(f"average faithfulness_score: {faithfulness_score_all/count}")
print(f"average completeness_score: {completeness_score_all/count}")
print(f"average conciseness_score: {conciseness_score_all/count}")
