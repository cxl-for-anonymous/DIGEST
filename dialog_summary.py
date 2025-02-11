import json
from LLM import create_open_llm
import re

llm = create_open_llm("")


class dialogue_node:
    def __init__(self, id, person, content, role, name):
        self.id = id
        self.person = person
        self.role = role
        self.name = name
        self.content = content
        self.children = []
        self.parent = None

    def add_child(self, id, ralation):
        self.children.append([id, ralation])


class dialogue_graph:
    def __init__(self):
        self.root = []
        self.id2node = {}

    def dfs(self, node_id):
        node = self.id2node[node_id]
        if (len(node.children) == 0):
            return [node_id]
        else:
            child_nodes = [node_id]
            for child in node.children:
                child_list = self.dfs(child[0])
                child_nodes += child_list
            return child_nodes


def event_collect(events):
    events_str = '\n'.join([f"{i + 1}. {event}" for i, event in enumerate(events)])
    prompt_event_summary = \
        """
        You are an AI assistant. Your task is to create a concise and insightful abstract by given sub-summaries.
        
        Please maintain consistency with the original summary in terms of vocabulary and try to retain precise wording as much as possible.
        
        These sub-summary fragments belong to a complete dialogue, please objectively connect the given sub-summaries to generate a summary, without any additional details or inferences.
        
        
        Sub-summaries:
        %s
        """ % (events_str)

    response = llm.predict(prompt_event_summary)

    prompt_reflection = \
        """
        You are an AI assistant designed to analyze and summarize dialogues.
        I'll give you a set of sub-summaries. At same time, I'll give you the aggregated summary of them.
        Your task is to modify the aggregated summary according to sub-summaries. Please follow the following steps:
        For each sentence in the summary:
        1. Check if there are any inconsistencies between the details in the sentence and that in sub-summaries (Focus on the accuracy of participants and actions). If so, correct them according to the events.
        2. Check if there are any details or inference are not mentioned in the sub-summaries, please also delete them.
        3. Check if there are any non-informative sub-summaries, such as greetings and small talk. If so, delete corresponding content in the aggregated summary.
        4. Check if there are any grammar error. If so, please correct it.
        
        Please output the result in the following format:
        [{"sentence": "content of sentence1", "thought": "reason for your determination", "corrected sentence": "corrected sentence1"}, ...]
        
        Please summarize the following text without including any headings like 'Summary:' or similar prefixes. Just provide the content of the summary.
        
        sub-summaries:
        %s
        
        aggregated summary:
        %s
        """ % (events_str, response)

    response_summary_final = llm.predict(prompt_reflection)

    try:
        response_str = response_summary_final.strip("```json").strip("```").strip()
        response = json.loads(response_str)
        summary = ""
        for item in response:
            summary += item["corrected sentence"] + " "
    except:
        summary = response
    return summary


def summary_sub_topic(dialogues, name_map):
    prompt_information_extract = \
        """
        You are an AI assistant designed to extract key points from dialogues.
        
        Your task is to analyze the given conversation, extract the key information points, and summarize them in a structured way.
        
        Events: Including a list of event. During the extraction process, please follow the original wording as closely as possible.
        For each event, please focus on the following two aspect: 
        Participants: Key person in an event, use the person id.
        Event description: What the participants did, the specific actions they took, the results of those actions, and the viewpoints expressed or information provided by the participants afterwards.
        Each event should be summarised as a sentence.
        
        Topic: This refers to the core topic of the conversation. 
        
        Please output in a json format data strcture. For example:
        {
            "Events":["event1", "event2", ...], 
            "Topic": "topic"
        }
        
        During the extraction process, please follow the original wording as closely as possible.
        
        The dialogues will be provided in the following format:
        Line ID. #Person ID#: dialogue content (reply to line Line ID)
        
        Dialogue sentences:
        %s
        """ % (dialogues)
    response_extract = llm.predict(prompt_information_extract)
    prompt_summary = \
        """
        You are an AI assistant. Your task is to summarize a given dialogue.
        Please ignore unnecessary details in the conversation and summarize in a concise way.
        
        Dialogue sentences:
        %s
        """ % (dialogues)

    response_summary = llm.predict(prompt_summary)
    prompt_reflection = \
        """
        You are an AI assistant designed to analyze and summarize dialogues.
        I'll give you a set of events extracted from dialogues, and the topic of the dialogues. At same time, I'll give you the summary of the dialogues.
        Your task is to modify the summary according to events and topics. Please follow the following steps:
        For each sentence in the summary:
        1. Check if there are any inconsistencies between the details in the sentence and that in events (Focus on the accuracy of participants and actions). If so, correct them according to the events.
         If it is not mentioned in the event, just ignore it, and consider it is correct.
        2. Check whether the content in the sentence that does not match the topic. If so, delete it. If the topic is empty, delete it too.
        
        Please output the result in the following format:
        [{"sentence": "content of sentence1", "thought": "reason for your determination", "corrected sentence": "corrected sentence1"}, ...]
        
        Please summarize the following text without including any headings like 'Summary:' or similar prefixes. Just provide the content of the summary.
        
        Events and Topic:
        %s
        
        summary:
        %s
        """ % (response_extract, response_summary)

    response_summary_final = llm.predict(prompt_reflection)

    response_str = response_summary_final.strip("```json").strip("```").strip()
    response = json.loads(response_str)
    try:
        summary = ""
        for item in response:
            summary += item["corrected sentence"] + " "
    except:
        summary = response_summary
    return summary


def sub_graph_summary(dia_graph, ids_triplet, name_map):
    ids = ids_triplet[0]
    ids.sort()

    dialogues = ""
    for id in ids:
        node = dia_graph.id2node[id]

        if (node.parent is not None and node.parent[1] == "Question-Answer Pair"):
            reply_attach = f" (reply to line {node.parent[0]})"
        else:
            reply_attach = ""

        dia_content = f"{node.id}. #{node.person}# : {node.content}{reply_attach}\n"
        if (node.content == ""):
            continue
        dialogues += dia_content
    if (dialogues == ""):
        return ""

    response = summary_sub_topic(dialogues, name_map)
    return response


def merge_consecutive_singles(list_of_lists):
    result = []
    temp_single_group = []
    for sublst in list_of_lists:
        if len(sublst) == 1:
            val = sublst[0]
            if not temp_single_group:
                temp_single_group.append(val)
            else:
                if val == temp_single_group[-1] + 1:
                    temp_single_group.append(val)
                else:
                    result.append(temp_single_group)
                    temp_single_group = [val]
        else:
            if temp_single_group:
                result.append(temp_single_group)
                temp_single_group = []
            result.append(sublst)
    if temp_single_group:
        result.append(temp_single_group)
    return result


import numpy as np
from sklearn.neighbors import KernelDensity


def find_high_density_region(lines, bandwidth=2.0, fraction=0.5):
    if not lines:
        return (None, None)
    X = np.array(lines)
    X.sort()
    X_reshaped = X.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X_reshaped)
    log_density_vals = kde.score_samples(X_reshaped)
    density_vals = np.exp(log_density_vals)
    max_density = np.max(density_vals)
    threshold = max_density * fraction
    filtered_points = X[density_vals >= threshold]
    if len(filtered_points) == 0:
        return (None, None)
    low, high = filtered_points[0], filtered_points[-1]
    return (low, high)


def get_mean(num_list):
    return sum(num_list) / len(num_list)


def is_in_range(value, rng):
    if rng is None or len(rng) != 2:
        return False
    low, high = rng
    if low is None or high is None:
        return False
    return low <= value <= high


def merge_datas(datas):
    merged_flag = True
    while merged_flag:
        merged_flag = False
        new_datas = []
        skip_indices = set()
        i = 0
        while i < len(datas):
            if i in skip_indices:
                i += 1
                continue
            has_merged = False
            j = i + 1
            while j < len(datas):
                if j in skip_indices:
                    j += 1
                    continue
                line_i, avg_i, range_i = datas[i]
                line_j, avg_j, range_j = datas[j]
                if is_in_range(avg_i, range_j) or is_in_range(avg_j, range_i):
                    merged_lines = line_i + line_j
                    merged_avg = get_mean(merged_lines)
                    merged_range = find_high_density_region(merged_lines)
                    new_datas.append([merged_lines, merged_avg, merged_range])
                    skip_indices.add(i)
                    skip_indices.add(j)
                    has_merged = True
                    merged_flag = True
                    break
                j += 1
            if not has_merged and i not in skip_indices:
                new_datas.append(datas[i])
            i += 1
        datas = new_datas

    return datas


def process_ids(num_list):
    merged = merge_consecutive_singles(num_list)
    datas = []
    for line in merged:
        avg = get_mean(line)
        high_range = find_high_density_region(line)
        datas.append([line, avg, high_range])
    final_datas = merge_datas(datas)
    return final_datas


def process(datas, key):
    try:
        print(key)
        data_line = datas[key]
        lines = data_line['parsing']
        speaker2role = data_line['roles']
        speaker2name = data_line['name_map']

        dia_graph = dialogue_graph()
        for dialogue_line in lines:
            line_id = dialogue_line[0]
            person = dialogue_line[1]
            content = dialogue_line[2]
            if (dialogue_line[3] is not None and dialogue_line[0] != 0 and dialogue_line[3][
                'related_line_id'] is not None):
                reply_to = int(dialogue_line[3]['related_line_id'])
                relation = dialogue_line[3]['relation']
            else:
                reply_to = None
                relation = None

            dia_node = dialogue_node(line_id, person, content, speaker2role[person], speaker2name[person])
            dia_graph.id2node[line_id] = dia_node
            if (reply_to is None):
                dia_graph.root.append(line_id)
            else:
                dia_graph.id2node[reply_to].add_child(line_id, relation)
                dia_graph.id2node[line_id].parent = [reply_to, relation]

        summaries = []
        ids_list = []
        for root_node in dia_graph.root:
            ids = dia_graph.dfs(root_node)
            ids.sort()
            ids_list.append(ids)
        ids_list_new = process_ids(ids_list)
        ids_list_new = sorted(ids_list_new, key=lambda x: x[1])
        for ids in ids_list_new:
            summary = sub_graph_summary(dia_graph, ids, speaker2role)
            if (len(re.findall("[a-zA-Z0-9]", summary)) > 0):
                summaries.append(summary)

        if (len(summaries) == 1):
            summary_final = summaries[0]
        else:
            summary_final = event_collect(summaries)
        for speaker in speaker2name:
            if (speaker2name[speaker]):
                summary_final = re.sub(speaker, f"{speaker} ({speaker2name[speaker][0]})", summary_final)
        return summary_final

    except Exception as e:
        print(e)
        print(f"{key} + error")


with open("data_processd.json", "r+") as f:
    datas = json.load(f)

results = []
for key in datas.keys():
    summary = process(datas, key)
    results.append(
        {"id": key, "dialogue": datas[key]['dialogue'], "summary": datas[key]['golden_summary'], "response": summary})

with open("data_evaluate.json", "w+") as f:
    json.dump(results, f, indent=4)

