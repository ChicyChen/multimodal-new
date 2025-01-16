import os
import numpy as np

root = '.'

start_list = []
end_list = []

for i in range(100):
    file = open(os.path.join(root, f'nw_train_tiny2048_{i+1}', 'training_logs.txt'), 'r')

    Lines = file.readlines()
    for line in Lines:
        text = line.strip()
        if text.find('Center distance: ') > -1:
            stepstart = text.find('Center distance: ') + len(('Center distance: '))
            center = float(text[stepstart:stepstart+len('1.4095')])
            if len(start_list) == len(end_list):
                start_list.append(center)
            else:
                end_list.append(center)
                break

start_list = np.array(start_list)
end_list = np.array(end_list)
changes = end_list - start_list

print(np.mean(changes))
np.save('changes.npy', changes)