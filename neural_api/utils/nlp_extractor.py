from neural_api.libs.json_lib import read_json, write_json
import os
from pathlib import Path

path = Path(__file__).resolve().parent.parent
INTENTS_PATH = "\\data\\intents\\"

def nlp_data_extractor() -> None:
    write_json(path + '\\work_data\\nlp_intent_data.json', "")
    q_intents = []
    a_responses = []
    for root, dirs, files in os.walk(path + INTENTS_PATH):
        for filename in files:
            try:
                if filename.find('usersays') != -1:
                    q_intents.append(filename)
                else:
                    a_responses.append(filename)
            except:
                ...

    data_in_file = {}
    for file in q_intents:
        data_list = []
        data_dict = {}
        file_data = read_json(path + INTENTS_PATH + file)

        for text_id in file_data:
            data_list.append(text_id['data'][0]['text'])

        data_dict["intent"] = data_list
        data_in_file[file] = data_dict

    a_data = []
    for file in a_responses:
        file_data = read_json(path + INTENTS_PATH + file)
        a_data.append(file_data["responses"][0]['messages'][0]['speech'])

    for i, file in enumerate(q_intents):
        data_in_file[file]["responses"] = a_data[i]

    write_json(path + "\\work_data\\nlp_intent_data.json", data_in_file, 'w')


if __name__ == '__main__':
    nlp_data_extractor()
