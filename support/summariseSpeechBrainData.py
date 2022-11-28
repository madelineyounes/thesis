import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np


ogfile = tuple(open('speechbrainTestData.csv', 'r'))
csvFile = open('newspeechbrainTestData.csv', 'w')
sumFile = open('speechbrainUmbrellaSummary.txt', 'w')
dialect_list = ['EGY', 'SDN', 'IRQ', 'KWT', 'ARE', 'QAT', 'OMN',
                'SAU', 'YEM', 'PSE', 'LBN', 'SYR', 'JOR', 'MRT', 'MAR', 'DZA', 'LBY']
umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']
summary = {}
summary['correct'] = 0
summary['total'] = 0
summary['langauges'] = []
i = 0
for line in ogfile:
    if (i % 2 != 0):
        filename = line.split(',')[0]
        dialect = line.split(',')[1].split('\n')[0]
    elif (i != 0):
        prediction = line.split(',')[1]
        lan = prediction.split("'")[1].split(':')[0]
        #csvFile.write(f"{filename},{dialect},{prediction}")
        if lan not in summary['langauges']:
            summary['langauges'].append(lan)

        if lan == 'ar':
            correct = 1
            summary['correct'] = summary.get('correct') + 1
        else:
            correct = 0

        if dialect in summary:
            item = summary[dialect]

            if lan in item.get('language'):
                item['language'][lan] = item['language'][lan] + 1
            else: 
                item['language'][lan] = 1

            item['total'] = item.get('total') + 1
            item['correct'] = item.get('correct') + correct
        else:
            language = {}
            language[lan] = 1
            summary[dialect] = {'correct': correct, 'total': 1, 'language' : language}
        summary['total'] = summary.get('total') + 1
    i+=1

def reg_to_umbr(summary):
    dialect_dict = {
        "EGY": ['EGY', 'SDN'],
        "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
        "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
        "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
    }

    newsummary = {}
    newsummary['correct'] = summary['correct']
    newsummary['total'] = summary['total']
    newsummary['langauges'] = summary['langauges']
    for u in dialect_dict:
        newsummary[u] = {}
        newsummary[u]["total"] = 0
        newsummary[u]["correct"] = 0
        newsummary[u]["language"] = {}
        for r in dialect_dict[u]:
            newsummary[u]["total"] += summary[r]["total"]
            newsummary[u]["correct"] += summary[r]["correct"]

            for lan in summary[r]["language"]:
                if lan in newsummary[u].get('language'):
                    newsummary[u]['language'][lan] += summary[r]['language'][lan]
                else:
                    newsummary[u]['language'][lan] = summary[r]['language'][lan]
    return newsummary

def dict_to_matrix(summary_dict, d_list): 
    lan_len = len(summary_dict["langauges"])
    matrix = np.empty((0, lan_len), dtype=int)
    i = 0 
    for d in d_list:
        row = np.array([])
        for l in summary_dict["langauges"]:
            if l in summary_dict[d]["language"]:
                row = np.append(
                    row, summary_dict[d]["language"][l]/summary_dict[d]["total"]*100)
            else:
                row = np.append(row, 0)
        if (i == 0):
            np.reshape(row, (1, lan_len)) 
            matrix = row

        else:
            np.reshape(row, (1, lan_len))
            matrix = np.vstack([matrix, row])
        i=+1
    return matrix 

def plot_data(x_label, y_label, matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)

    fig.colorbar(cax)
    xaxis = np.arange(len(x_label))
    yaxis = np.arange(len(y_label))
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    plt.show()

def gen_report(mysummary, outfile, d_list):
    print("SUMMARY REPORT")
    outfile.write("SUMMARY REPORT\n")
    for d in d_list:
        d_sum = mysummary[d]["language"]
        my_keys = sorted(d_sum, key=d_sum.get, reverse=True)[:3]
        print(f"DIALECT: {d}")
        outfile.write(f"DIALECT: {d}\n")
        print("Language \t files/total \t pct")
        outfile.write("Language \t files/total \t pct\n")
        for k in my_keys:
            f = mysummary[d]["language"][k]
            tot = mysummary[d]["total"]
            per = (f/tot)*100
            print(f"{k} \t\t {f}/{tot} \t {per:.2f}%")
            outfile.write(f"{k} \t\t {f}/{tot} \t {per:.2f}%\n")

    print("OVERALL SUMMARY")
    outfile.write("OVERALL SUMMARY")
    tot = summary["total"]
    cor = summary["correct"]
    per = cor/tot * 100
    print(f"{cor}/{tot} \t {per:.2f}%")
    outfile.write(f"{cor}/{tot} \t\t {per:.2f}%\n")


umbrella = reg_to_umbr(summary)
gen_report(umbrella, sumFile, umbrella_dialects)
summary_matrix = dict_to_matrix(umbrella, umbrella_dialects)
plot_data(umbrella['langauges'], umbrella_dialects, summary_matrix)


