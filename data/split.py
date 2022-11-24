import os
import shutil
import pandas as pd


# tsvファイルを読み込む
data_csv = pd.read_csv('./train_master.tsv', sep='\t')

# emotion(感情)に該当する列を取り出す
label_list = data_csv['expression'].tolist()
id_list = data_csv['id'].tolist()

# 画像を指定のフォルダに分ける
data_path = './train'
data_label_path = './train/label'
for id_, label in enumerate(label_list):
    src_path = os.path.join(data_path, id_list[id_])
    dst_path = os.path.join(data_label_path, label)
    shutil.copy(src_path, dst_path) # コピーする
