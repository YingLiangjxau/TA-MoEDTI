import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Code'))
from main import main

# 配置区域 - 修改这里
ORIGINAL_DATASETS = ['biosnap','bindingdb','drugbank']

CROSS_DATASETS = [
    'DRUGBANK_WO_BIOSNAP',
    'DRUGBANK_WO_BINDINGDB',
    'BIOSNAP_WO_DRUGBANK',
    'BIOSNAP_WO_BINDINGDB',
    'BINDINGDB_WO_DRUGBANK',
    'BINDINGDB_WO_BIOSNAP'
]
DATABASES = ORIGINAL_DATASETS + CROSS_DATASETS
HIDDEN_CHANNELS = 64
REPEAT_TIMES = 20


# 执行
for db in CROSS_DATASETS:
        for i in range(REPEAT_TIMES):
            print(f"Running: {db}, hidden_channels={HIDDEN_CHANNELS}, repeat={i+1}")
            main(DATABASE=db, hidden_channels=HIDDEN_CHANNELS)
