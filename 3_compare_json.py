import os 
import json
import numpy as np

dir1 = "/home/ubuntu/ashok/Backtesting/trt_outputs/production_model"
dir2 = "/home/ubuntu/ashok/Backtesting/trt_outputs/final_2"



#do all files exist : yes
# set1 = set(os.listdir(dir1))
# set2 = set(os.listdir(dir2))
# print(set1-set2)
# print(set2-set1)


def extract_values(json):
    phone,gaze = None,None
    for item in json:
        if item['name'] == "probs_phone":
            phone = item['values']
        elif item['name'] == "probs_gaze":
            gaze = item['values']
    return phone,gaze

file_names = []
phone1_list = []
phone2_list = []
gaze1_list = []
gaze2_list = []

for file in os.listdir(dir1):
    file_names.append(file)
    with open(os.path.join(dir1,file), 'r') as f1:
        json1 = json.load(f1)
    with open(os.path.join(dir2,file), 'r') as f2:
        json2 = json.load(f2)

    phone1,gaze1 = extract_values(json1)
    phone2,gaze2 = extract_values(json2)

    phone1_list.append(phone1)
    phone2_list.append(phone2)
    gaze2_list.append(gaze2)
    gaze1_list.append(gaze1)

phone_diff = (np.array(phone1_list) - np.array(phone2_list)).max()
gaze_diff = (np.array(gaze1_list) - np.array(gaze2_list)).max()

phone_diff_flat = phone_diff.flatten()
gaze_diff_flat = gaze_diff.flatten()

# import pandas as pd
# df = pd.DataFrame({'phone':phone_diff_flat,
#                     'gaze':gaze_diff_flat})
# df.describe()

def statistics(diff):
    mean = np.mean(diff)
    std = np.std(diff)
    min_ = np.min(diff)
    max_ = np.max(diff)
    median = np.median(diff)
    percentile_25 = np.percentile(diff, 25)
    percentile_75 = np.percentile(diff, 75)

    print(f"""mean: {mean}
std: {std}, min: {min_}
max: {max_}, 
median: {median}, 
25th percentile: {percentile_25}, 
75th percentile: {percentile_75}""")

print('phone:')
statistics(phone_diff_flat)

print('gaze:')
statistics(gaze_diff_flat)

# [
#   { "name" : "probs_phone"
#   , "dimensions" : "1x5"
#   , "values" : [ 0.0435101, 0.0340782, 0.0555826, 0.195444, 0.671385 ]
#   }
# , { "name" : "probs_headset"
#   , "dimensions" : "1x3"
#   , "values" : [ 0.79014, 0.206794, 0.00306536 ]
#   }
# , { "name" : "probs_gaze"
#   , "dimensions" : "1x5"
#   , "values" : [ 0.0187541, 0.0112795, 0.0027413, 0.00542685, 0.961798 ]
#   }
# ]

    


