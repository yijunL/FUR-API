# coding: UTF-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from config import Config
from preprocess import DataProcessor, get_time_dif
from model import CNNAEModel
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, f1_score
import json
from CNNEncoder import CNNSentenceEncoder

parser = argparse.ArgumentParser(description="API Anomaly Detection")
parser.add_argument("--mode", type=str, default="train", help="train/test")
parser.add_argument("--data_dir", type=str, default="./data", help="training data and saved model path")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
args = parser.parse_args()


reconstruction_criterion = nn.MSELoss()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, config, train_iterator, dev_iterator):
    model.train()
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    t_total = len(train_iterator) * config.num_epochs
    break_flag = False
    total_batch = 0
    last_improve = 0
    best_dev_loss = float('inf')

    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        with open("logger.txt", "a+", encoding="utf-8") as fw:
            fw.write("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))

        for _, (batch_data, labels) in enumerate(train_iterator):
            decoded, cls_vector = model(
                input_ids=batch_data)

            reconstruction_loss = reconstruction_criterion(decoded, cls_vector)
            reconstruction_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if total_batch % config.log_batch == 0:
                dev_loss = eval(model, dev_iterator)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), config.saved_model)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Batch Train Loss: {1:>5.2}, Val Loss: {2:>5.2}, Time: {3} {4}'
                print(msg.format(total_batch, reconstruction_loss.item(), dev_loss, time_dif, improve))
                with open("logger.txt", "a+", encoding="utf-8") as fw:
                    fw.write(msg.format(total_batch, reconstruction_loss.item(), dev_loss, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No improvement for a long time, auto-stopping...")
                break_flag = True
                break
        if break_flag:
            break


def eval(model, iterator):
    model.eval()
    re_loss = 0.00
    with torch.no_grad():
        for batch, labels in iterator:
            decoded, cls_vector = model(
                input_ids=batch)

            re_loss = re_loss + reconstruction_criterion(decoded, cls_vector)
        re_loss = re_loss / len(iterator)
    return re_loss

def threshold(model, config, iterator,auto_rate = 0.25):
    model.eval()
    model.load_state_dict(torch.load(config.saved_model))
    loss_set = []
    with torch.no_grad():
        for batch, labels in iterator:
            decoded, cls_vector = model(
                input_ids=batch)

            reconstruction_loss = reconstruction_criterion(decoded, cls_vector)
            loss_set.append(reconstruction_loss.item())

    loss_set = sorted(loss_set)
    return loss_set[int(len(loss_set)*auto_rate)]

def test(model, config, iterator, flag=0.5):
    print("testing...")
    model.eval()
    model.load_state_dict(torch.load(config.saved_model))
    start_time = time.time()
    pre_label = []
    tru_label = []
    for batch, labels in iterator:
        decoded, cls_vector = model(
            input_ids=batch)

        reconstruction_loss = reconstruction_criterion(decoded, cls_vector)
#         print("test reconstruction_loss: ", reconstruction_loss.item())
        with open("logger.txt", "a+", encoding="utf-8") as fw:
            fw.write("test reconstruction_loss: " + str(reconstruction_loss.item()))
        # 0:normal  1:abnormal
        if reconstruction_loss.item() <= flag:
            pre_label.append(0)
        else:
            pre_label.append(1)
        tru_label.append(int(labels.cpu()))

    print(tru_label)
    print(pre_label)
    auc = roc_auc_score(tru_label, pre_label)
    recall = recall_score(tru_label, pre_label)
    tn, fp, fn, tp = confusion_matrix(tru_label, pre_label).ravel()
    fpr = fp / (fp + tn)
    f1 = f1_score(tru_label, pre_label)

    msg = "AUC: {0:>5.5},  F1-score: {1:>5.5},  Recall: {2:>5.5}, fpr: {3:>5.5}"
    print(msg.format(auc, f1, recall, fpr))
    with open("logger.txt", "a+", encoding="utf-8") as fw:
        fw.write(msg.format(auc, f1, recall, fpr))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def main():
    set_seed(args.seed)
    config = Config(args.data_dir)
    
    max_length = config.max_seq_len
    glove_mat = np.load('./pretrain/glove.6B.50d_mat.npy')
    glove_word2id = json.load(open('./pretrain/glove.6B.50d_word2id.json'))

    cnn_encoder = CNNSentenceEncoder(
            glove_mat,
            glove_word2id,
            max_length)
    cnn_encoder.to(config.device)
    
    
    model = CNNAEModel(cnn_encoder)
    model.to(config.device)
    

    if args.mode == "train":
        print("loading data...")
        start_time = time.time()
        train_iterator = DataProcessor(config.train_file, config.device, cnn_encoder, config.batch_size,
                                       config.max_seq_len, args.seed)
        dev_iterator = DataProcessor(config.dev_file, config.device, cnn_encoder, config.batch_size,
                                     config.max_seq_len, args.seed)
        test_iterator = DataProcessor(config.test_file, config.device, cnn_encoder, 1, config.max_seq_len,
                                      args.seed)
        
        anomaly_iterator = DataProcessor(config.train_anomaly_file, config.device, cnn_encoder, 1,
                                       config.max_seq_len, args.seed)
        time_dif = get_time_dif(start_time)
        print("time usage:", time_dif)

        # train
        train(model, config, train_iterator, dev_iterator)
        t = threshold(model, config, anomaly_iterator)
        print("Auto-threshold: ",t)
        # test
        test(model, config, test_iterator, flag = t)
    elif args.mode == "test":
        print("loading data...")
        start_time = time.time()
        anomaly_iterator = DataProcessor(config.train_anomaly_file, config.device, cnn_encoder, 1,
                                       config.max_seq_len, args.seed)
        test_iterator = DataProcessor(config.test_file, config.device, cnn_encoder, 1, config.max_seq_len,
                                      args.seed)
        time_dif = get_time_dif(start_time)
        print("time usage:", time_dif)
        # test
        t = threshold(model, config, anomaly_iterator)
        print("Auto-threshold: ",t)
        test(model, config, test_iterator, flag = t)


if __name__ == '__main__':
    main()
