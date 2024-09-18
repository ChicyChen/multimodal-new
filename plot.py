import matplotlib.pyplot as plt 
import os

# root_list = [".3_shrink_nw_logs_1_5e-4", "3_shrink_nw_logs_1e-1_5e-4", "3_shrink_nw_logs_1e-2_5e-4", "3_shrink_nw_logs_5e-1_5e-4"]
root_list = ["3_shrink_tiny_nw_train_checkpoints_learn_5e-4_1e-1"]
for root in root_list:
    # Using readlines()
    file1 = open(os.path.join(root, 'training_logs.txt'), 'r')
    Lines = file1.readlines()
    plt.figure()
    steps_list = []
    loss_list = []
    # Strips the newline character
    for line in Lines:
        text = line.strip()
        prevlen = len('2024-04-03 17:19:29,360 ')
        interlen1 = len('CLIP_COCO_TRAIN INFO: Epoch:')
        lastlen = len(' (5.5949)')
        stepstart = text.find('global_step') + len(('global_step: '))
        stepend = text.find(', lr:')
        if text[prevlen: prevlen+interlen1] == 'CLIP_COCO_TRAIN INFO: Epoch:':
            # print(text)
            step_num = int(text[stepstart:stepend])
            loss = float(text[-lastlen-6:-lastlen])
            steps_list.append(step_num)
            loss_list.append(loss)


    plt.plot(steps_list, loss_list)
    plt.title('Loss versus steps')
    plt.savefig(os.path.join(root, 'loss.png'))