"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: record_utils.py
    @Time: 2025/5/21 22:09
    @Email: None
"""

import json
import os
import tempfile

import pandas as pd



def make_train_path(record_path, model_name, start_time):

    time_path = os.path.join(record_path, model_name, start_time)
    best_path = os.path.join(record_path, model_name, start_time, 'best_result')
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)
    return time_path, best_path


def package_one_epoch(**kwargs):
    one_epoch_data = json.dumps(kwargs)
    return one_epoch_data


def save_train_data(target_path, start_time, end_time, list_data, top_data):

    records = [json.loads(item) for item in list_data]
    df = pd.DataFrame(records)
    excel_name = 'Trian.xlsx'

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    excel_path = os.path.join(target_path, excel_name)
    df.to_excel(str(excel_path), index=False, sheet_name='Train_data')


    time_file = f'{start_time}-{end_time}.txt'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    time_file_path = os.path.join(target_path, time_file)
    with open(time_file_path, 'w', encoding='utf-8') as f:
        f.write(f'{start_time}-{end_time}')
    f.close()


    json_name = 'Trian.json'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    json_path = os.path.join(target_path, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    f.close()


    top_path = os.path.join(target_path, 'best_result', 'top_result.json')
    with open(top_path, 'w', encoding='utf-8') as f:
        json.dump(top_data, f, ensure_ascii=False, indent=4)
    f.close()
    return excel_path, json_path, top_path


def save_train_config(save_path, **kwargs):
    json_data = json.dumps(kwargs, indent=4, ensure_ascii=False)
    file_path = os.path.join(save_path, 'Train config.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    f.close()
    return file_path

def record_model_description(json_file, model_name, model_description):

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if model_name in data:
        return

    data[model_name] = model_description


    dir_name = os.path.dirname(json_file)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, indent=4, ensure_ascii=False)
        temp_name = tmp_file.name


    os.replace(temp_name, json_file)