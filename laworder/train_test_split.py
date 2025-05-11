# -*- coding: utf-8 -*-
import json
import random
from collections import defaultdict
from collections import Counter

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def split_train_dev_test(data, label_col, train_ratio=0.7, dev_ratio=0.2, test_ratio=0.1):
    # 1. 레이블별로 데이터를 그룹화하기 위한 딕셔너리 생성
    label_groups = defaultdict(list)

    for item in data:
        label = item[label_col]  # 레이블이 있는 필드 이름에 맞게 수정 필요
        label_groups[label].append(item)

    train_data, dev_data, test_data = [], [], []

    # 2. 각 레이블 그룹을 나눠서 train, dev, test로 분할
    for label, items in label_groups.items():
        random.shuffle(items)  # 데이터를 섞음
        n_total = len(items)

        # 각 비율에 맞게 데이터 개수를 계산
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)
        n_test = n_total - n_train - n_dev  # 나머지는 test로 할당

        # 데이터를 분리하여 각 리스트에 추가
        train_data.extend(items[:n_train])
        dev_data.extend(items[n_train:n_train + n_dev])
        test_data.extend(items[n_train + n_dev:])

    return train_data, dev_data, test_data

def split_train_dev(data, label_col, train_ratio, dev_ratio):
    # 1. 레이블별로 데이터를 그룹화하기 위한 딕셔너리 생성
    label_groups = defaultdict(list)

    for item in data:
        label = item[label_col]  # 레이블이 있는 필드 이름에 맞게 수정 필요
        label_groups[label].append(item)

    train_data, dev_data = [], []

    # 2. 각 레이블 그룹을 나눠서 train, dev 로 분할
    for label, items in label_groups.items():
        random.shuffle(items)  # 데이터를 섞음
        n_total = len(items)

        # 각 비율에 맞게 데이터 개수를 계산
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)

        # 데이터를 분리하여 각 리스트에 추가
        train_data.extend(items[:n_train])
        dev_data.extend(items[n_train:n_train + n_dev])

    return train_data, dev_data

def split_data_by_output_source(data, train_ratio):
    # 1. output과 source에 따른 그룹화
    grouped_data = defaultdict(list)

    for key in data.keys():
        item = data[key]
        # output과 source를 키로 사용하여 그룹화
        key = (item['output'], item['source'])
        grouped_data[key].append(item)

    train_data, dev_data = [], []
    # 2. 각 그룹별로 데이터를 나누기
    for key, items in grouped_data.items():

        random.shuffle(items)  # 데이터를 무작위로 섞음
        n_total = len(items)

        # 각 비율에 따라 데이터 개수 결정
        n_train = int(n_total * train_ratio)

        # 데이터를 각각의 세트로 나눔
        train_data.extend(items[:n_train])
        dev_data.extend(items[n_train:])

    return train_data, dev_data

def check_label_statistics_by_source(data):


    all_labels = []
    for t_k in data.keys():
        #print(raw_dataset_test[t_k])
        all_labels.append(data[t_k]['source'])

    frequency = Counter(all_labels)
    return frequency

def make_t1_split_dataset():
    """
    split 기준 : vishing label => 2017~2022.10까지의 데이터는 train, dev로, 2022.11부터의 데이터는 test 데이터로
    non_vishing label => 7:2:1로 랜덤하게 분리
    """

    COLUMNS = ['conversation', 'time', 'source', 'output']

    with open('../data/task1/task1_v1.8.json', 'r') as file:
        raw_dataset = json.load(file)

    print(len(raw_dataset))

    # (1) 2022.11부터의 vishing 데이터 + 같은 수의 non_vishing 데이터를 test로 만들고 기존 딕셔너리에서 제거
    raw_dataset_test = defaultdict(list)
    # (1-1) non_vishing인 데이터 test 개수만큼(1,428개) 찾아서 기록
    for data_source in ['real_crime_investigator', 'counselor', 'national']:

        globals()['non_vishing_key_for_{}'.format(data_source)] = [key for key in raw_dataset.keys() if raw_dataset[key]['source'] == data_source]
        globals()['non_vishing_for_test_{}'.format(data_source)] = defaultdict(list)

        for conv_key in globals()['non_vishing_key_for_{}'.format(data_source)]:
            globals()['non_vishing_for_test_{}'.format(data_source)][conv_key] = raw_dataset[conv_key]

        random_keys = random.sample(list(globals()['non_vishing_for_test_{}'.format(data_source)].keys()), 476)
        for key in random_keys:
            raw_dataset_test[key] = globals()['non_vishing_for_test_{}'.format(data_source)][key]
            del raw_dataset[key]

    # (1-2) vishing 데이터 중 2022.11~2023 데이터 찾아서 기록
    time_info = []
    #print(len(raw_dataset))
    for test_key in raw_dataset.keys():

        time = raw_dataset[test_key]['time']
        #print(time)
        #key = raw_dataset.keys()[i]
        output = raw_dataset[test_key]['output']
        try:
            time = str(time).split('-')[0] + '-' + str(time).split('-')[1]
        except:
            time = str(time).split('-')[0]

        if '2022-11' in time or '2022-12' in time or '2023' in time:
            raw_dataset_test[test_key] = raw_dataset[test_key]
            del raw_dataset[test_key]

    print(len(raw_dataset))
    print(len(raw_dataset_test))

    raw_dataset_train_list, raw_dataset_dev_list = split_data_by_output_source(raw_dataset, train_ratio=0.78)
    raw_dataset_train = defaultdict(list)
    raw_dataset_dev = defaultdict(list)

    for i in range(len(raw_dataset_train_list)):
        raw_dataset_train[i] = raw_dataset_train_list[i]

    for j in range(len(raw_dataset_dev_list)):
        raw_dataset_dev[j] = raw_dataset_dev_list[j]

    print(len(raw_dataset_train))
    print(len(raw_dataset_dev))

    print(check_label_statistics_by_source(raw_dataset_train))
    print(check_label_statistics_by_source(raw_dataset_dev))
    print(check_label_statistics_by_source(raw_dataset_test))

    with open('../data/task1/task1_v1.8_train.json', 'w') as json_file:
        json.dump(raw_dataset_train, json_file, ensure_ascii=False, indent=4)

    with open('../data/task1/task1_v1.8_dev.json', 'w') as json_file:
        json.dump(raw_dataset_dev, json_file, ensure_ascii=False, indent=4)

    with open('../data/task1/task1_v1.8_test.json', 'w') as json_file:
        json.dump(raw_dataset_test, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    make_t1_split_dataset()