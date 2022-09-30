import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


ogfile = tuple(open('speechbrainTestData.csv', 'r'))
csvFile = open('newspeechbrainTestData.csv', 'w')
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
print(summary)



