import json
from LLM import create_open_llm
import concurrent.futures

llm = create_open_llm("")

def name_extract(context, roles):
    prompt_name_extract = \
"""
You are an expert in conversational analysis. Your task is to extract all names and titles mentioned in the given conversation history. 
This includes actual names (e.g., 'John', 'Alice').
Note that any titles, roles or speaker ids (e.g., 'Mom', 'Doctor', 'Manager', 'Person A') are not included in the names.

The conversation is in the following format:
Speaker id : What he/she said.

The output should be a JSON-formatted summary that clearly outlines each name. The format should look like this: 
["name1", "name2", ...]

Please provide all names mentioned in the conversation history.

Conversation history: %s

Please only output the names in JSON format, and do not output any other information.
""" % (context)
    response_str = llm.predict(prompt_name_extract)

    response_str = response_str.strip("```json").strip("```").strip()
    response = json.loads(response_str)

    if(response==[]):
        speaker2name_map = {}
        roles = roles.replace("'", "\"")
        roles_list = json.loads(roles)
        for role in roles_list:
            speaker2name_map[role] = []
        return speaker2name_map

    names = []
    for name in response:
        if(name not in names):
            names.append(name)

    speaker2name_map = {}
    roles = roles.replace("'", "\"")
    roles_list = json.loads(roles)
    for id in roles_list:
        speaker2name_map[id] = []
    for name in names:
        if(name in roles_list):
            continue
        if(name not in context):
            continue
        prompt_name_align = \
"""
You are an expert in conversational analysis. Your task is to align the names with the speaker IDs based on the given conversation history, IDs, and names. You will match the name or title to the corresponding speaker ID based on how they appear in the conversation history.

The output should be a JSON-formatted summary that clearly shows the mapping between speaker IDs and their corresponding names. The output must contain the following information:

name: The given name (e.g., 'John').
thought: Why you believe the name corresponds to the speaker id. First, why you think 'name' is a real name rather than a role or a title. Then you have to give the dialogue where 'name' is called and the dialogue where corresponding speaker replied. Following that, make a summaries who is also called 'name' or no one is called 'name' in speaker ids.
speaker id: The ID of the speaker (e.g., 'SpeakerA').

The format should look like this: 
{"name": "John", "thought": "'John' is a real name rather than a title and it can be inferred from the conversation that he is someone present. SpeakerB called name 'John' in dialogue 'SpeakerB : John let's go', and Speaker A replied to it in dialogue 'SpeakerA : Okey'. So SpeakerA is also called 'John'.", "speaker id": "SpeakerA"}
If it is not a real name, please reply empty str "" in speaker id like this:
{"name": "Mom", "thought": "Mom is a familial title, not a name, as it signifies a relationship rather than an individual identity.", "speaker id": ""}

Given the following inputs:
speaker ids: %s
name: %s
conversation history: %s
Please align the speaker IDs with their corresponding name and output the result in JSON format. Do not include any other information.""" % (roles, name, context)
        response_str = llm.predict(prompt_name_align)
        try:
            response_str = response_str.strip("```json").strip("```").strip()
            response = json.loads(response_str)
            if(response['speaker id'] == ""):
                continue
            speaker2name_map[response["speaker id"]].append(response["name"])
        except:
            continue

    return speaker2name_map

def role_extract(context, roles):
    prompt_role_extract = \
"""
You are an expert in conversational analysis and role identification.

Your task is to analyze the given conversation history and determine the roles each speaker is playing according to the conversation context, using as few words as possible to describe each role.

Output format: Provide a JSON-formatted summary that clearly outlines the specific role or function each participant has in this conversation. The format should be as follows:

[{"name": "Speaker A","role": "Role description"},{"name": "Speaker B","role": "Role description"}]

Conversation history: %s

Please identify the roles of the following individuals separately: %s
""" % (context, roles)
    response_str = llm.predict(prompt_role_extract)

    response_str = response_str.strip("```json").strip("```").strip()
    response = json.loads(response_str)
    speaker2role_map = {}
    for line in response:
        speaker2role_map[line["name"]] = line["role"]

    return speaker2role_map


def get_reply_id(history_dialogue, current_dialogue):
    prompt_reply=\
"""
You are a dialogue analysis assistant, skilled at identifying interaction relationships between speakers within multi-turn dialogues and accurately recognizing relationships in conversation.
Your task is to: Given a history of dialogue records and a current dialogue line, identify the relationship between the current dialogue line and a line in the historical dialogue. 
You need to determine the connection between the two lines, considering the content, context, speaker roles, and previous interactions to accurately identify the type of interaction between them.

Data Format:
History dialogue record format: line id. [Speaker] (Speaker's other name (if exists)) (Speaker Role): Dialogue content.
Current dialogue format: [Speaker] (Speaker's other name (if exists)) (Speaker Role): Dialogue content.

Here are some types of conversational relationships you can refer to.
Direct Reply: The current conversation record is a direct response to a previous one, typically answering a question or responding to a specific statement.
Indirect Reply: The current record refers to a previous topic or event without directly replying to a specific record, continuing the discussion in a subtle way.
Discussion Continuation: The current record extends or adds to a previous topic, usually with a time gap but still part of the same discussion thread.
Information Provision: The current record provides new information or clarifies a point from a previous record, advancing the conversation.
Contradiction or Refutation: The current record contradicts or challenges a previous statement, often offering a different opinion or perspective.
Confirmation: The current record confirms or agrees with a previous statement or opinion, reinforcing a shared viewpoint.
Supplement/Extension: The current record supplements or adds detail to a previous record, providing more context or relevant information.
New Topic: The current record introduces a new topic that is not directly related to any previous record but shifts the conversation in a new direction.
Topic Pause or Shift: The current record temporarily changes or pauses the topic, possibly steering the conversation in a different direction or relieving tension.

Thinking Steps:
1.Examine whether the current dialogue references or responds to any line in the history dialogue by analyzing expressions, context, and keyword associations.
2.Consider the influence of speaker roles on the relationship, such as hierarchical or role-based dynamics within the dialogue.
3.Understand the logical connection between the current dialogue and historical dialogue, determining whether it is a response, continuation, or a new topic.
4.Select the most relevant historical dialogue line as the reply target for the current dialogue.

Output Format:
Provide the output in JSON format, including:
"reasoning": A brief description of the reasoning process, including evidence from the corresponding line and content. If "related_line_id" is None, explain why no relevant line could be found.
"related_line_id": The line id in the history that the current dialogue is replying to (or None if no relevant line is found).
"relation":  A concise description of the relationship between the two lines (e.g., "Direct Reply", "Indirect Reference", "new topic"). If "related_line_id" is None, indicate "no relation".

History dialogue:
%s

Current dialogue:
%s
""" % (history_dialogue, current_dialogue)
    try:

        response_str = llm.predict(prompt_reply)

        response_str = response_str.strip("```json").strip("```").strip()
        response = json.loads(response_str)
    except:
        response = {"relation":None, "reasoning":"llm error", "related_line_id":None}
    return response




def process(datas, id):
    line = datas[id]
    print(f"{id} begin")
    try:
        conversation_lines = line["utterance"]
        context = ""
        roles = set()
        for i in range(len(conversation_lines)):
            cline = conversation_lines[i]
            roles.add(cline[1])
            context += f"{cline[1]} : {cline[2]}\n"

        roles_str = str(list(roles))

        try:
            speaker2role_map = line["roles"]
        except:
            speaker2role_map = role_extract(context, roles_str)

        speaker2name_map = name_extract(context, roles_str)

        line["roles"] = speaker2role_map
        line["names"] = speaker2name_map

        results=[]
        history_dialogue = ""
        for j in range(len(conversation_lines)):
            cline = conversation_lines[j]
            if(j==0):
                results.append([j, cline[1], cline[2], None])
                history_dialogue += f"{j}. [{cline[1]}] ({speaker2name_map[cline[1]]}) ({speaker2role_map[cline[1]]}) : {cline[2]}\n"
                continue

            current = f"[{cline[1]}] ({speaker2name_map[cline[1]]}) ({speaker2role_map[cline[1]]}) : {cline[2]}\n"

            response = get_reply_id(history_dialogue, current)

            results.append([j, cline[1], cline[2], response])

            history_dialogue+=f"{j}. {current}"
        print(f"{id} done")
        return id,results
    except:
        print(f"{id} error")
        return []

