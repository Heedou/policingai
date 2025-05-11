import os
import json
import pandas as pd
from datasets import Dataset, concatenate_datasets


class RawDataset(object):
    def __init__(self, conf, logging):
        self.path_rawdata = os.path.join(conf.path.dataset, conf.dataprep.raw_dataset)
        self.path_ftdata  = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.logging  = logging


        self.subsample   = conf.dataprep.subsample

        self.logging.info(f"[SUBSAMPLE]                       :         {self.subsample}")

        self.ft_dataset = conf.dataprep.finetuning_dataset

    def make_t8_instruction_dataset(self):

        COLUMNS = ['conversation', 'output']

        with open(self.path_rawdata + "/task8_v1_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task8_v1_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task8_v1_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['output'] = raw_dataset[key]['output']

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 범죄 관련 텍스트를 읽고 미끼문자의 수법을 분류할 수 있습니다. 주어진 텍스트를 자세히 읽고, 주어진 메시지가 어떤 미끼문자 수법에 해당하는 지 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
입력 : "범죄관련 텍스트"
분석 결과 : "해당하는 미끼문자 수법 한 개."
---
입력 : {x['conversation']}
분석 결과 : {x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task8_v1_{key_split}_{self.subsample}.hf")

    def make_t6_instruction_dataset(self):

        COLUMNS = ['conversation', 'source', 'output', 'input_tokens', 'ner_tokens', 'output_with_ner_tokens']

        with open(self.path_rawdata + "/task6_v2_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task6_v2_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task6_v2_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['output_with_ner_tokens'] = ''
                for i, j in zip(raw_dataset[key]['input_tokens'], raw_dataset[key]['ner_tokens']):
                    if j != 'o':
                        dict_data['output_with_ner_tokens'] += i+f'[{j}] '
                    else:
                        dict_data['output_with_ner_tokens'] += i + ' '
                print(dict_data['output_with_ner_tokens'])
                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 범죄 관련 텍스트를 읽고 범죄 정보의 분석에 필요한 범죄 개체명을 추출할 수 있습니다. 주어진 텍스트를 자세히 읽고, 아래 형식에 맞추어 범죄 개체명을 추출하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
입력 : "범죄관련 텍스트"
분석 결과 : "주어진 입력 텍스트를 공백을 기준으로 토큰화 한 뒤, 추출된 토큰과 인식된 범죄 개체명을 제시. 공백 기준으로 구분된 토큰들의 BIO 태깅 방식에 의한 추출."
---
입력 : {x['conversation']}
분석 결과 : {x['output_with_ner_tokens']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task6_v2_{key_split}_{self.subsample}.hf")

    def make_t5_crimename_instruction_dataset(self):

        COLUMNS = ['conversation', 'crime_name']

        with open(self.path_rawdata + "/task5_v3_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['crime_name'] = "\n".join(raw_dataset[key]['crime_name'])

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 범죄 관련 텍스트를 읽고 한국의 형사법률에 따른 죄명을 분석할 수 있습니다. 주어진 텍스트를 자세히 읽고, 범인의 행위가 어떤 죄명에 해당하는지 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
입력 : "범죄관련 텍스트"
분석 결과 : "해당하는 죄명 한 개."
---
입력 : {x['conversation']}
분석 결과 : {x['crime_name']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task5_v3_crimename_{key_split}_{self.subsample}.hf")

    def make_t5_law_instruction_dataset(self):

        COLUMNS = ['conversation', 'law']

        with open(self.path_rawdata + "/task5_v3_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['law'] = "\n".join(raw_dataset[key]['law'])

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 범죄 관련 텍스트를 읽고 한국의 형사법률에 따른 위반 법률 조항을 분석할 수 있습니다. 주어진 텍스트를 자세히 읽고, 범인의 행위가 어떤 법률 조항에 해당하는지 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
입력 : "범죄관련 텍스트"
분석 결과 : "해당하는 형사법률조항 나열. 여러 개 일 수 있음"
---
입력 : {x['conversation']}
분석 결과 : {x['law']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task5_v3_law_{key_split}_{self.subsample}.hf")

    def make_t5_component_instruction_dataset(self):

        COLUMNS = ['conversation', 'component']

        with open(self.path_rawdata + "/task5_v3_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task5_v3_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['component'] = "\n".join(raw_dataset[key]['component'])

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 범죄 관련 텍스트를 읽고 한국의 형사법률에 따른 구성요건적 행위를 분석할 수 있습니다. 주어진 텍스트를 자세히 읽고, 텍스트의 내용에 반영된 범인의 구성요건적 행위가 무엇인지 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
입력 : "범죄관련 텍스트"
분석 결과 : "구성요건적 행위 분석 결과를 나열. 여러 개 일 수 있음"
---
입력 : {x['conversation']}
분석 결과 : {x['component']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task5_v3_component_{key_split}_{self.subsample}.hf")

    def make_t2_instruction_dataset(self):

        COLUMNS = ['conversation', 'intention', 'file', 'case', 'time']

        with open(self.path_rawdata + "/task2_v4_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/task2_v4_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/task2_v4_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['intention'] = "\n".join(raw_dataset[key]['intention'])
                dict_data['time'] = raw_dataset[key]['time']
                dict_data['file'] = raw_dataset[key]['file']
                dict_data['case'] = raw_dataset[key]['case']

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 보이스피싱 대화를 읽고 범인의 발언 의도를 분석할 수 있습니다. 주어진 대화를 자세히 읽고, 보이스피싱 범죄 스크립트 분석기법에 따라 보이스피싱 사기 범죄의 특징을 찾은 후 범인의 발화 의도가 무엇인지 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
대화 : "보이스피싱에 해당하는 대화 내용"
분석 결과 : "범인의 발화 의도를 분류"
---
대화 : {x['conversation']}
분석 결과 : {x['intention']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task2_v4_{key_split}_{self.subsample}.hf")

    def make_next_utterance_instruction_dataset(self):

        COLUMNS = ['conversation', 'output', 'file', 'case', 'time']

        with open(self.path_rawdata + "/next_utterance_v1_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata + "/next_utterance_v1_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata + "/next_utterance_v1_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'),
                                       (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['output'] = raw_dataset[key]['output']
                dict_data['file'] = raw_dataset[key]['file']
                dict_data['case'] = raw_dataset[key]['case']
                dict_data['time'] = raw_dataset[key]['time']

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 대화를 읽고 보이스피싱 범인과 시민의 대화 내용을 이해할 수 있습니다. 주어진 대화를 자세히 읽고, 범죄 스크립트 분석에 따를 때 범인이 대화 다음에 시민을 속이기 위해 주로 사용할 발언에 대해서 예측해 보세요.
        ---
        항상 다음과 같은 형식을 따라야 합니다.
        대화 : "대화 내용. 보이스피싱 범인과 시민 간의 대화 일부"
        분석 결과 : "주어진 대화 이후의 예상되는 범인의 주요 발언들"
        ---
        대화 : {x['conversation']}
        분석 결과 : {x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/next_utterance_v1_{key_split}_{self.subsample}.hf")

    def make_korccvi_instruction_dataset(self):

        COLUMNS = ['conversation','output']

        raw_dataset_train = pd.read_csv('data/KorCCVi/KorCCViD_v1.3_train.csv')
        raw_dataset_dev = pd.read_csv('data/KorCCVi/KorCCViD_v1.3_dev.csv')

        data_instances = dict(train=[], dev=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev')]:
            for key in range(len(raw_dataset)):
                dict_data = dict()
                dict_data['conversation'] = raw_dataset['Transcript'][key]
                if raw_dataset['Label'][key] == 0:
                    dict_data['output'] = 'non_vishing'
                elif raw_dataset['Label'][key] == 1:
                    dict_data['output'] = 'vishing'

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))

            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 대화를 읽고 보이스피싱 범행을 위한 대화인지 아닌지를 구분할 수 있습니다. 주어진 대화를 자세히 읽고, 실제 수사관의 출석요구이거나 실제 금융기관의 상담 대화 일수도 있는 점을 감안하여, 보이스피싱 사기 범죄의 특징을 찾아 분석하세요.
        ---
        항상 다음과 같은 형식을 따라야 합니다.
        대화 : "대화 내용. 보이스피싱일 수 도 있고, 아닐 수도 있음"
        분석 결과 : "vishing or non_vishing."
        ---
        대화 : {x['conversation']}
        분석 결과 : {x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/KorCCVi_v1.3_{key_split}_{self.subsample}.hf")

    def make_t1_instruction_dataset(self):

        COLUMNS = ['conversation', 'time', 'source', 'output']

        with open(self.path_rawdata+"/task1_v1.8_train.json", 'r') as f:
            data = f.read()
        raw_dataset_train = json.loads(data)

        with open(self.path_rawdata+"/task1_v1.8_dev.json", 'r') as f:
            data = f.read()
        raw_dataset_dev = json.loads(data)

        with open(self.path_rawdata+"/task1_v1.8_test.json", 'r') as f:
            data = f.read()
        raw_dataset_test = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])

        for raw_dataset, key_split in [(raw_dataset_train, 'train'), (raw_dataset_dev, 'dev'), (raw_dataset_test, 'test')]:
            for key in raw_dataset.keys():
                dict_data = dict()
                dict_data['conversation'] = raw_dataset[key]['conversation']
                dict_data['time'] = raw_dataset[key]['time']
                dict_data['source'] = raw_dataset[key]['source']
                dict_data['output'] = raw_dataset[key]['output']

                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 대화를 읽고 보이스피싱 범행을 위한 대화인지 아닌지를 구분할 수 있습니다. 주어진 대화를 자세히 읽고, 실제 수사관의 출석요구이거나 실제 금융기관의 상담 대화 일수도 있는 점을 감안하여, 보이스피싱 사기 범죄의 특징을 찾아 분석하세요.
---
항상 다음과 같은 형식을 따라야 합니다.
대화 : "대화 내용. 보이스피싱일 수 도 있고, 아닐 수도 있음"
분석 결과 : "vishing or non_vishing."
---
대화 : {x['conversation']}
분석 결과 : {x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/task1_v1.8_{key_split}_{self.subsample}.hf")


    def make_expert_explain_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            dict_data = dict()
            dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
            dict_data['year']              = year
            dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id']
            dict_data['subject']           = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data['output'] = raw_dataset[key]['expert_explain']
                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:

            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
    f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
    ---

    항상 다음과 같은 형식을 따라야 합니다.

    법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

    추론 결과 : "참 또는 거짓. 법률 가설의 판단. "

    ---

    법률 가설 : {x['hypothesis']}
    추론 결과 : {x['output']}<|endoftext|>""" })

            data.save_to_disk(f"{self.path_ftdata}/explain_{key_split}_{self.subsample}.hf")


    def make_expert_correct_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer'] # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1,7)]:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_expert_corrected' 
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
       f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
        ---
    
        항상 다음과 같은 형식을 따라야 합니다.
    
        법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
        전제 : "5개의 전제 사실"
        전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
        추론: "법률 가설의 판단을 위한 추론 과정"
    
        답변: "법률 가설의 판단. 참 또는 거짓."
    
        ---
    
        법률 가설: {x['hypothesis']}
    
        전제: {x['premise']}
    
        추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/correct_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/correct_{key_split}_{self.subsample}.hf")

    def make_correct_explain_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        data_instances_explain = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_explain']
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_'  + '_expert_explain'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_explain[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data_explain = Dataset.from_pandas(pd.DataFrame(data=data_instances_explain[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))

                data = data.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
         ---
    
         항상 다음과 같은 형식을 따라야 합니다.
    
         법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
         전제 : "5개의 전제 사실"
         전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
         추론: "법률 가설의 판단을 위한 추론 과정"
    
         답변: "법률 가설의 판단. 참 또는 거짓."
    
         ---
    
         법률 가설: {x['hypothesis']}
    
         전제: {x['premise']}
    
         추론 결과:{x['output']}<|endoftext|>"""})

                data_explain = data_explain.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
                ---
    
                항상 다음과 같은 형식을 따라야 합니다.
    
                법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
                추론: "법률 가설의 판단을 위한 추론 과정"
    
                답변: "법률 가설의 판단. 참 또는 거짓."
    
                ---
    
                법률 가설 : {x['hypothesis']}
                추론 결과 : {x['output']}<|endoftext|>"""})

                dataset_cc = concatenate_datasets([data, data_explain])

                dataset_cc.save_to_disk(f"{self.path_ftdata}/correct_explain_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data_test = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))

                data_test.save_to_disk(f"{self.path_ftdata}/correct_explain_{key_split}_{self.subsample}.hf")

    def make_only_3s_setting_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise


            dict_data = dict()
            dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
            dict_data['premise'] = raw_dataset[key]['premise']
            dict_data['output'] = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
            # dict_data['output_corrected']  = None
            dict_data['year'] = year
            dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id']
            dict_data['subject'] = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
            data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))

            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
     ---

     항상 다음과 같은 형식을 따라야 합니다.

     법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

     전제 : "5개의 전제 사실"
     전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

     추론: "법률 가설의 판단을 위한 추론 과정"

     답변: "법률 가설의 판단. 참 또는 거짓."

     ---

     법률 가설: {x['hypothesis']}

     전제: {x['premise']}

     추론 결과:{x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/only_3s_setting_{key_split}_{self.subsample}.hf")


    def make_expert_curation_only_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise


            dict_data = dict()
            dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
            dict_data['premise'] = raw_dataset[key]['premise']
            if raw_dataset[key]['expert_correction'] != 'nan' and raw_dataset[key]['expert_correction'] != 'None':
                dict_data['output'] = raw_dataset[key]['expert_correction']
            else:
                dict_data['output'] = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
            # dict_data['output_corrected']  = None
            dict_data['year'] = year
            dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id']
            dict_data['subject'] = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
            data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))

            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
     ---

     항상 다음과 같은 형식을 따라야 합니다.

     법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

     전제 : "5개의 전제 사실"
     전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

     추론 결과: "참 또는 거짓. 법률 가설의 판단을 위한 추론 과정"

     ---

     법률 가설: {x['hypothesis']}

     전제: {x['premise']}

     추론 결과:{x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/expert_curation_only_{key_split}_{self.subsample}.hf")

    def make_6s_rationales_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer'] # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])
        data_instance_test = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1,7)]:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)
            if year in self.years_test:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id']
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instance_test[key_split].append(dict_data) #희두 추가


        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
       f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
        ---
    
        항상 다음과 같은 형식을 따라야 합니다.
    
        법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
        전제 : "5개의 전제 사실"
        전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
        추론: "법률 가설의 판단을 위한 추론 과정"
    
        답변: "법률 가설의 판단. 참 또는 거짓."
    
        ---
    
        법률 가설: {x['hypothesis']}
    
        전제: {x['premise']}
    
        추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/6s_rationales_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instance_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/6s_rationales_{key_split}_{self.subsample}.hf")

    def make_simple_finetuning_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']  # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
                                   f"""주어진 전제에 의할 때, 법률 가설은 참인가 거짓인가?
                                   
                                   전제: {x['premise']}
                                   
                                   법률 가설: {x['hypothesis']}
                                   
                                   추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/simple_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/simple_{key_split}_{self.subsample}.hf")

    def make_6s_solution_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        data_instances_explain = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_test[key_split].append(dict_data)
            except:
                pass

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_explain']
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + '_expert_explain'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_explain[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data_explain = Dataset.from_pandas(pd.DataFrame(data=data_instances_explain[key_split],
                                                                columns=COLUMNS).sample(frac=self.subsample))

                data = data.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
         ---

         항상 다음과 같은 형식을 따라야 합니다.

         법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

         전제 : "5개의 전제 사실"
         전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

         추론: "법률 가설의 판단을 위한 추론 과정"

         답변: "법률 가설의 판단. 참 또는 거짓."

         ---

         법률 가설: {x['hypothesis']}

         전제: {x['premise']}

         추론 결과:{x['output']}<|endoftext|>"""})

                data_explain = data_explain.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
                ---

                항상 다음과 같은 형식을 따라야 합니다.

                법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

                추론: "법률 가설의 판단을 위한 추론 과정"

                답변: "법률 가설의 판단. 참 또는 거짓."

                ---

                법률 가설 : {x['hypothesis']}
                추론 결과 : {x['output']}<|endoftext|>"""})

                dataset_cc = concatenate_datasets([data, data_explain])

                dataset_cc.save_to_disk(f"{self.path_ftdata}/6s_solution_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data_test = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                             columns=COLUMNS).sample(frac=self.subsample))

                data_test.save_to_disk(f"{self.path_ftdata}/6s_solution_{key_split}_{self.subsample}.hf")