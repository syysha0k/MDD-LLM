import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_csv(csv_path):
    assert os.path.exists(csv_path)
    csv_rec = pd.read_csv(csv_path, low_memory=False)
    return csv_rec


def generate_id_prompt(csv_rec, ratio=1.0):
    prompt_line = []
    for idx in tqdm(range(csv_rec.shape[0])):
        temp_line = csv_rec.iloc[idx]
        # 23 feature names
        sentence_list = []

        if temp_line['Year'] is not np.NaN:
            temp_01_age = 'age is {}'.format(2010 - temp_line['Year'])
            sentence_list.append(temp_01_age)

        if temp_line['Sex'] is not np.NaN:
            temp_02_sex = 'sex is {}'.format(temp_line['Sex'].lower())
            sentence_list.append(temp_02_sex)

        if temp_line['BMI0'] is not np.NaN:
            temp_03_bmi = 'body mass index (BMI) is {} kg/m^2'.format(temp_line['BMI0'])
            sentence_list.append(temp_03_bmi)

        if temp_line['Sleeplessness'] is not np.NaN:
            temp_item = temp_line['Sleeplessness']
            if '/' in temp_item:
                temp_item = temp_item.split('/')[0]
            temp_04_sleeplessness = '{} sleeplessness'.format(temp_item.lower())
            sentence_list.append(temp_04_sleeplessness)

        if temp_line['Sleep'] is not np.NaN:
            if not temp_line['Sleep'].isdigit():
                continue
            if int(temp_line['Sleep']) > 1:
                temp_05_sleeptime = 'sleep time is {} hours'.format(temp_line['Sleep'])
            else:
                temp_05_sleeptime = 'sleep time is {} hour'.format(temp_line['Sleep'])
            sentence_list.append(temp_05_sleeptime)

        if temp_line['Alcohol frequency'] is not np.NaN:
            temp_item = temp_line['Alcohol frequency'].lower()
            if 'never' in temp_item:
                temp_06_alcohol = '{} drink alcohol'.format(temp_item)
            else:
                temp_06_alcohol = 'drink alcohol {}'.format(temp_item)
            sentence_list.append(temp_06_alcohol)

        if temp_line['self-harmed'] is not np.NaN:
            temp_item = temp_line['self-harmed'].lower()
            if 'yes' in temp_item:
                temp_07_harmed = 'has self-harming behavior'
            elif 'no' in temp_item:
                temp_07_harmed = 'never self-harmed'
            else:
                raise ValueError('** Not support {} @ 07'.format(temp_item))
            sentence_list.append(temp_07_harmed)

        if temp_line['suicide'] is not np.NaN:
            temp_item = temp_line['suicide'].lower()
            if 'yes' in temp_item:
                temp_08_suicide = 'has attempted suicide'
            elif 'no' in temp_item:
                temp_08_suicide = 'has never suicide'
            else:
                raise ValueError('** Not support {} @ 08'.format(temp_item))
            sentence_list.append(temp_08_suicide)

        if temp_line['Mental problems diagnosed'] is not np.NaN:
            temp_09_mental = 'has mental problems: {}'.format(temp_line['Mental problems diagnosed'].lower())
            sentence_list.append(temp_09_mental)

        if temp_line['Happiness'] is not np.NaN:
            temp_10_happiness = 'often feel {}'.format(temp_line['Happiness'].lower())
            sentence_list.append(temp_10_happiness)

        if temp_line['Work/job satisfaction'] is not np.NaN:
            temp_11_work_sat = 'the job satisfaction is {}'.format(temp_line['Work/job satisfaction'].lower())
            sentence_list.append(temp_11_work_sat)

        if temp_line['employment status'] is not np.NaN:
            if 'None of the above' == temp_line['employment status']:
                continue
            temp_12_employment = 'the employment status is {}'.format(temp_line['employment status'].lower())
            sentence_list.append(temp_12_employment)

        if temp_line['income'] is not np.NaN:
            temp_13_income = 'the income is {} dollar'.format(temp_line['income'])
            sentence_list.append(temp_13_income)

        if temp_line['Length of working week'] is not np.NaN:
            temp_14_work_time = 'work {} hours per week'.format(temp_line['Length of working week'])
            sentence_list.append(temp_14_work_time)

        if temp_line['Qualifications'] is not np.NaN:
            if 'None of the above' == temp_line['Qualifications']:
                continue
            temp_15_edu = 'the education is {}'.format(temp_line['Qualifications'].lower())
            sentence_list.append(temp_15_edu)

        if temp_line['Long-standing illness'] is not np.NaN:
            temp_item = temp_line['Long-standing illness'].lower()
            if 'yes' in temp_item:
                temp_16_long_ill = 'has long-standing illness'
            elif 'no' in temp_item:
                temp_16_long_ill = 'not has long-standing illness'
            else:
                continue
            sentence_list.append(temp_16_long_ill)

        if temp_line['Health satisfaction'] is not np.NaN:
            temp_item = temp_line['Health satisfaction'].lower()
            temp_17_health_sat = 'the health satisfaction is {}'.format(temp_item)
            sentence_list.append(temp_17_health_sat)

        if temp_line['Family relationship satisfaction'] is not np.NaN:
            temp_item = temp_line['Family relationship satisfaction'].lower()
            temp_18_fr_sat = 'the family relationship satisfaction is {}'.format(temp_item)
            sentence_list.append(temp_18_fr_sat)

        if temp_line['Financial situation satisfaction'] is not np.NaN:
            temp_item = temp_line['Financial situation satisfaction'].lower()
            temp_19_fs_sat = 'the financial situation satisfaction is {}'.format(temp_item)
            sentence_list.append(temp_19_fs_sat)

        if temp_line['HDLC-Blood0'] is not np.NaN:
            temp_20_hdlc = 'the HDL cholesterol is {} mmol/L'.format(temp_line['HDLC-Blood0'])
            sentence_list.append(temp_20_hdlc)

        if temp_line['CLDLC0'] is not np.NaN:
            temp_21_cldlc = 'the Clinical LDL Cholesterol is {} mmol/L'.format(temp_line['CLDLC0'])
            sentence_list.append(temp_21_cldlc)

        if temp_line['TG0'] is not np.NaN:
            temp_22_tg = 'the Triglycerides is {} mmol/L'.format(temp_line['TG0'])
            sentence_list.append(temp_22_tg)

        if temp_line['TC0'] is not np.NaN:
            temp_23_tc = 'the Total Cholesterol is {} mmol/L'.format(temp_line['TC0'])
            sentence_list.append(temp_23_tc)

        # add shuffle element
        # random.shuffle(sentence_list)

        temp_sentence = ','.join(sentence_list)
        prompt_line.append(temp_sentence)

    return prompt_line


def write_text(prompt_line, save_path):
    save_rec = open(save_path, 'w')
    for line in prompt_line:
        save_rec.write(line)
        save_rec.write('\n')

    save_rec.close()


def build_json(prompt_lines, labels=None, json_path=None):
    # with open(json_path, 'w') as file_j:
    temp_mdd_train, temp_hc_train = [], []
    temp_mdd_test, temp_hc_test = [], []
    json_path_train, json_path_test = json_path

    file_train = open(json_path_train, 'w')
    file_test = open(json_path_test, 'w')

    prompt_line_mdd, prompt_line_hc = prompt_lines
    label_mdd, label_hc = labels

    for idx, line in tqdm(enumerate(prompt_line_mdd)):
        line = line.capitalize()
        # temp_dict = {
        #     "instruction": "Predict if a patient has the major depressive disorder? Yes or no? Please answer with only yes or no and do not give any extra information.",
        #     "input": line,
        #     "output": label_mdd,
        # }
        label_mdd_temp = '{},{}%'.format(label_mdd, str(prob_mdd))

        temp_dict = {
            "instruction": "Predict if a patient has the major depressive disorder? Yes or no? "
                           "Please answer with yes or no and give corresponding probability."
                           "Please answer exactly in the format below, without blank lines and no "
                           "further information or answer is required.Please answer exactly in the format below, "
                           "without blank lines and no further information or answer is required. "
                           "{Yes or No},{in percentages, round to two decimal place}"
                           "For example: No,82.32%",
            "input": line,
            "output": label_mdd_temp,
        }

        if idx % 10 == 0:
            temp_mdd_test.append(temp_dict)
        else:
            temp_mdd_train.append(temp_dict)


    # generate hc json
    for idx, line in tqdm(enumerate(prompt_line_hc_select)):
        line = line.capitalize()
        # temp_dict = {
        #     "instruction": "Predict if a patient has the major depressive disorder? Yes or no? Please answer with only yes or no and do not give any extra information.",
        #     "input": line,
        #     "output": label_hc,
        # }
        label_hc_temp = '{},{}%'.format(label_hc, str(prob_hc))

        temp_dict = {
            "instruction": "Predict if a patient has the major depressive disorder? Yes or no? "
                           "Please answer with yes or no and give corresponding probability. "
                           "Please answer exactly in the format below, without blank lines and no "
                           "further information or answer is required.Please answer exactly in the format below, "
                           "without blank lines and no further information or answer is required. "
                           "{Yes or No},{in percentages, round to two decimal place}"
                           "For example: No,82.32%",
            "input": line,
            "output": label_hc_temp,
        }
        # temp_dict = {
        #     "instruction": "Predict if a patient has the major depressive disorder? Yes or no? Please answer with only yes or no and corresponding probability.",
        #     "input": line,
        #     "output": label_hc,
        # }
        if idx % 10 == 0:
            temp_hc_test.append(temp_dict)
        else:
            temp_hc_train.append(temp_dict)

    temp_list_train = temp_mdd_train + temp_hc_train
    temp_list_test = temp_mdd_test + temp_hc_test

    # add some test sample for training
    temp_list_test_select_train = random.sample(temp_list_test, int(0.000001 * len(temp_list_test)))
    temp_list_train_v2 = temp_list_train + temp_list_test_select_train
    # temp_list_train_v2 = temp_list_train

    # add mapping
    # for idx_text in tqdm(temp_list_test):
    #     for idx_train in temp_list_train_v2:
    #         temp_res = idx_text['input'] == idx_train['input']
    #         print(idx_text['input'], idx_train['input'])
    #         if temp_res:
    #             print('Error')
    #             exit(0)


    # json.dump(temp_list_train, file_train, indent=2)
    json.dump(temp_list_train_v2, file_train, indent=2)
    json.dump(temp_list_test, file_test, indent=2)

    file_train.close()
    file_test.close()
    # json.dump(temp_hc, file_j, indent=2)

    print('==> Finish Build Json File.')


if __name__ == '__main__':
    csv_path_mdd = './data_save/extract_mdd.csv'
    csv_path_hc = './data_save/extract_hc.csv'

    ratio = 0.8
    json_path_train = './data_save/ratio/prompt_all_train_{}.json'.format(ratio)
    json_path_test = './data_save/ratio/prompt_all_test_{}.json'.format(ratio)

    csv_rec_mdd = load_csv(csv_path_mdd)
    prompt_line_mdd = generate_id_prompt(csv_rec_mdd, ratio=ratio)
    csv_rec_hc = load_csv(csv_path_hc)
    prompt_line_hc = generate_id_prompt(csv_rec_hc, ratio=ratio)

    build_json(prompt_lines=[prompt_line_mdd, prompt_line_hc], labels=['Yes', 'No'],
               json_path=[json_path_train, json_path_test])
