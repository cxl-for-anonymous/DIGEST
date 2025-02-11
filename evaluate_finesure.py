from evaluation.finesure.utils import get_fact_checking_prompt, parsing_llm_fact_checking_output, \
    get_keyfack_list_prompt, get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output, \
    compute_conciseness_percentage_score, compute_completeness_percentage_score, compute_faithfulness_percentage_score
import json
import re
from LLM import create_open_llm

with open("", "r+") as f:
    data_generated = json.load(f)

llm = create_open_llm("")

def process(line):
    try:
        if("faithfulness_score" in line.keys()):
            return line

        generated_summary = line['response']
        for i in range(1,10):
            generated_summary = generated_summary.replace(f'Person{i}', f'#Person{i}#')
        generated_summary = generated_summary+" "
        generated_summary_list_ = generated_summary.split(". ")
        generated_summary_list = []
        for g in generated_summary_list_:
            if(len(re.findall("[a-zA-Z0-9]", g))>0):
                generated_summary_list.append(g)
        golden_summary = line['summary']
        dialogue = line['dialogue']
        prompt_fact_checking = get_fact_checking_prompt(dialogue, generated_summary_list)
        fact_checking_str = llm.predict(prompt_fact_checking)
        pred_faithfulness_labels, pred_faithfulness_error_type = parsing_llm_fact_checking_output(fact_checking_str)
        prompt_keyfact_generate = get_keyfack_list_prompt(golden_summary)
        key_fact_list_str = llm.predict(prompt_keyfact_generate)
        key_fact_list = json.loads(key_fact_list_str)
        prompt_align = get_keyfact_alighment_prompt(keyfacts=key_fact_list['key facts'],
                                                    sentences=generated_summary_list)
        align_str = llm.predict(prompt_align)
        pred_alignment_labels, pred_sentence_line_numbers = parsing_llm_keyfact_alighment_output(align_str)

        faithfulness_score = compute_faithfulness_percentage_score(pred_faithfulness_labels)
        completeness_score = compute_completeness_percentage_score(pred_alignment_labels)
        conciseness_score = compute_conciseness_percentage_score(pred_sentence_line_numbers,
                                                                 len(generated_summary_list))

        line['fact_checking_str'] = fact_checking_str
        line['key_fact_list_str'] = key_fact_list_str
        line['align_str'] = align_str
        line['faithfulness_score'] = faithfulness_score
        line['completeness_score'] = completeness_score
        line['conciseness_score'] = conciseness_score
        print(f"{line['id']} + done")
        return line
    except:
        print(f"{line['id']} + error")
        return line

results = []
for line in data_generated:
    line_new = process(line)
    results.append(line_new)

with open("", "w+") as f:
    json.dump(results, f, indent=4)

