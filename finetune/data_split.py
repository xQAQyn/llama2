from torch.utils.data import random_split
import json

## Load the origin
def load_data():
    datas = []
    with open("./data/piqa_train.txt","r") as dialogfile, open("data/train_labels.txt","r") as labelfile:
        for dialog, label in zip(dialogfile, labelfile):
            temp_dialog = json.loads(dialog)
            data = {
                "instruction" : "Please respond with the number of the correct option",
                "input": (
                    f"question: {temp_dialog['goal']}?\n"
                    f"option 0: {temp_dialog['sol1']}\n"
                    f"option 1: {temp_dialog['sol2']}"
                ),
                "output": label.strip()
            }
            datas.append(data)
    return datas

datas = load_data()

train_rate = 0.8
train_size = int(len(datas) * train_rate)
val_size = len(datas) - train_size
train_data, val_data = random_split(datas, [train_size, val_size])

with open("./finetune/data/fine_tune_train.txt","w") as trainfile:
    for data in train_data:
        trainfile.write(json.dumps(data) + "\n")

with open("./finetune/data/fine_tune_val.txt","w") as valfile:
    for data in val_data:
        valfile.write(json.dumps(data) + "\n")