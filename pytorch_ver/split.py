import os
import shutil
import pandas as pd



# tsvファイルを読み込む
train_data_csv = pd.read_csv('./data/train_master.tsv', sep='\t')

# emotionに該当する列を取り出す
train_label_list = train_data_csv['expression'].tolist()
train_img_list = train_data_csv['id'].tolist()


train_data_path = './data/train'
train_data_label_path = './data/train/label'
# 画像を指定のフォルダに分ける
for id_ in range(len(train_label_list)):
    src_path = os.path.join(train_data_path, train_img_list[id_])
    dst_path = os.path.join(train_data_label_path, train_label_list[id_])
    shutil.copy(src_path, dst_path)





