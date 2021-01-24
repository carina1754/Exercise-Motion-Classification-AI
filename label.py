import pandas as pd
import matplotlib.pyplot as plt
import os

img_path = 'figs' #이미지를 저장할 디렉토리

train_x = pd.read_csv('train_features.csv') #train feature load
train_y = pd.read_csv('train_labels.csv') #label load

#label은 dictionary로 만들어 놓으면 좋을 것 같음
label_dict = dict()
for label, label_desc in zip(train_y.label, train_y.label_desc):
    label_dict[label] = label_desc
label_dict[45] ='Squat (kettlebell , goblet)'

for target_label in range(len(label_dict)):
    print(label_dict[target_label])
    target_ids = train_y[(train_y.label ==target_label)].id.to_numpy()
    temp_label_path = os.path.join(img_path,label_dict[target_label])
    os.mkdir(temp_label_path)
    for num, id_ in enumerate(target_ids):
        temp_features = train_x.loc[train_x.id == id_, 'acc_x': 'gy_z']
        plt.figure(figsize = (10,10))
        temp_features.plot()
        plt.savefig(os.path.join(temp_label_path,label_dict[target_label]+str(num)))
        plt.clf()
        plt.close('all')
