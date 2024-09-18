import matplotlib.pyplot as plt 
import os


plot_loss = False
plot_deri = False


root_list = ["train_checkpoints_temp7e-2", "train_checkpoints_temp7e-2_increase", "train_checkpoints_temp7e-2_learn"]
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
        if plot_loss:
            stepend = text.find(', lr:')
        elif plot_deri:
            stepend = text.find(', derivative:')
        else:
            stepend = text.find(', temp:')

        if text[prevlen: prevlen+interlen1] == 'CLIP_COCO_TRAIN INFO: Epoch:':
            if stepend == -1:
                pass 
            else:
                step_num = int(text[stepstart:stepend])
                steps_list.append(step_num)
                if plot_loss:
                    loss = float(text[-lastlen-6:-lastlen])
                    loss_list.append(loss)
                elif plot_deri:
                    loss = float(text[stepend+len(', derivative:'):-1])
                    loss_list.append(loss)
                else:
                    loss = float(text[stepend+len(', temp:'):-1])
                    loss_list.append(loss)

    plt.plot(steps_list, loss_list)
    if plot_loss:
        plt.title('Loss versus steps')
        plt.savefig(os.path.join(root, 'loss.png'))
    elif plot_deri:
        plt.title('Norm(Derivative) versus steps')
        plt.savefig(os.path.join(root, 'deri.png'))
    else:
        plt.title('Temp versus steps')
        plt.savefig(os.path.join(root, 'temp.png'))