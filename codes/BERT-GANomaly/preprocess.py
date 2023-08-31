# coding: UTF-8

import time
import torch
import random
from tqdm import tqdm
from datetime import timedelta
import pandas as pd

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class DataProcessor(object):
    def __init__(self, path, device, tokenizer, batch_size, max_seq_len, seed):
        self.seed = seed
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        # 加载数据
        self.data = self.load(path)
        self.index = 0
        # residue 剩余
        self.residue = False
        self.num_samples = len(self.data[0])
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.residue = True

    def load(self, path):
        contents = []
        labels = []
        data = pd.read_csv(path, encoding="utf-8")
        max = -1000
        count = 0
        for index, row in data.iterrows():
            label = row["type"]
#             content = "method:" + str(row["method"]).strip() + " request_body:" + str(row["request_body"]).strip() + " request_url:"  \
#                       + str(row["request_url"]).strip() + " response_body:" + str(row["response_body"]).strip() + " source_ip:" +  \
#                       str(row["source_ip"]).strip() + " source_port:" + str(row["source_port"]).strip() + " status:" +  str(row["status"]).strip() + \
#                       " target_ip:" + str(row["target_ip"]).strip() + " target_port:" + str(row["target_port"]).strip() + " response_time:" + \
#                       str(row["response_time"]).strip() + " user_identity:" + row["user_identity"].strip()
            content = "method:" + str(row["method"]).strip() + " request_body:" + str(row["request_body"]).strip() + " request_url:"  \
                      + str(row["request_url"]).strip() + " response_body:" + str(row["response_body"]).strip() + " status:" +  str(row["status"]).strip() + " response_time:" + \
                      str(row["response_time"]).strip() + " user_identity:" + row["user_identity"].strip()
            contents.append(content)
            if len(content) > 512:
                count = count + 1
            # print(content)
            if len(content) > max:
                max = len(content)
            # 0:normal  1:abnormal
            if label == "normal":
                labels.append(0)
            elif label == "anomaly":
                labels.append(1)
            else:
                print("label error!\n")
                break
        print("max: ", max)
        print("len > 512 count: ", count)
        # random shuffle
        index = list(range(len(labels)))
        random.seed(self.seed)
        random.shuffle(index)

        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        # print(contents)
        # print(labels)
        return contents, labels

    def __next__(self):
        if self.residue and self.index == self.num_batches:
            batch_x = self.data[0][self.index * self.batch_size: self.num_samples]
            batch_y = self.data[1][self.index * self.batch_size: self.num_samples]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch
        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            batch_x = self.data[0][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch_y = self.data[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch

    def _to_tensor(self, batch_x, batch_y):
        inputs = self.tokenizer.batch_encode_plus(
            batch_x,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation="longest_first",
            return_tensors="pt")
        inputs = inputs.to(self.device)
        labels = torch.LongTensor(batch_y).to(self.device)
        return inputs, labels

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches
