# coding: UTF-8

import time
import torch
import argparse
import numpy as np
from transformers import BertTokenizer
import torch.nn as nn
from config import Config
from preprocess import DataProcessor, get_time_dif
from model import BertVAEModel
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, f1_score

parser = argparse.ArgumentParser(description="API Anomaly Detection")
parser.add_argument("--mode", type=str, default="test", help="train/test")
parser.add_argument("--data_dir", type=str, default="./data", help="training data and saved model path")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
args = parser.parse_args()

# 损失函数
reconstruction_criterion = nn.MSELoss()  # 均方误差损失


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, config, train_iterator, dev_iterator):
    model.train()
    start_time = time.time()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    t_total = len(train_iterator) * config.num_epochs
    break_flag = False
    total_batch = 0
    last_improve = 0
    best_dev_loss = float('inf')
    # 训练循环
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        with open("logger.txt", "a+", encoding="utf-8") as fw:
            fw.write("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # 开始训练
        for _, (batch_data, labels) in enumerate(train_iterator):
            x, decoded, mu, logvar = model(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'])
            # 计算重构损失
            loss = model.compute_loss(x, decoded, mu, logvar)
            loss.backward()
            # 更新模型参数
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
                print(msg.format(total_batch, loss.item(), dev_loss, time_dif, improve))
                with open("logger.txt", "a+", encoding="utf-8") as fw:
                    fw.write(msg.format(total_batch, loss.item(), dev_loss, time_dif, improve))
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
            x, decoded, mu, logvar = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
            # 计算重构损失
            re_loss = re_loss + model.compute_loss(x, decoded, mu, logvar)
        re_loss = re_loss / len(iterator)
    return re_loss


def threshold(model, config, iterator, auto_rate=0.25):
    model.eval()
    model.load_state_dict(torch.load(config.saved_model))
    loss_set = []
    with torch.no_grad():
        for batch, labels in iterator:
            x, decoded, mu, logvar = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
            # 计算重构损失
            reconstruction_loss = model.compute_rec_loss(x, decoded, mu, logvar)
            loss_set.append(reconstruction_loss.item())

    loss_set = sorted(loss_set)
    return loss_set[int(len(loss_set) * auto_rate)]


def test(model, config, iterator, flag=0.5):
    # 在测试集上进行预测和评估
    print("testing...\n")
    model.eval()
    model.load_state_dict(torch.load(config.saved_model))
    start_time = time.time()
    pre_label = []
    tru_label = []
    for batch, labels in iterator:
        x, decoded, mu, logvar = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"])
        # 计算重构损失
        reconstruction_loss = model.compute_rec_loss(x, decoded, mu, logvar)
        #         print("test reconstruction_loss: ", reconstruction_loss.item())
        with open("logger.txt", "a+", encoding="utf-8") as fw:
            fw.write("test reconstruction_loss: " + str(reconstruction_loss.item()))
            fw.write("\n")

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
        fw.write("\n")

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



def main():
    set_seed(args.seed)
    config = Config(args.data_dir)
    tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
    model = BertVAEModel()
    model.to(config.device)

    if args.mode == "train":
        print("loading data...")
        start_time = time.time()
        train_iterator = DataProcessor(config.train_file, config.device, tokenizer, config.batch_size,
                                       config.max_seq_len, args.seed)
        dev_iterator = DataProcessor(config.dev_file, config.device, tokenizer, config.batch_size,
                                     config.max_seq_len, args.seed)
        test_iterator = DataProcessor(config.test_file, config.device, tokenizer, 1, config.max_seq_len,
                                      args.seed)
        
        anomaly_iterator = DataProcessor(config.train_anomaly_file, config.device, tokenizer, 1,
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
        anomaly_iterator = DataProcessor(config.train_anomaly_file, config.device, tokenizer, 1,
                                       config.max_seq_len, args.seed)
        test_iterator = DataProcessor(config.test_file, config.device, tokenizer, 1, config.max_seq_len,
                                      args.seed)
        time_dif = get_time_dif(start_time)
        print("time usage:", time_dif)
        # test
        t = threshold(model, config, anomaly_iterator)
        print("Auto-threshold: ",t)
        test(model, config, test_iterator, flag = t)


if __name__ == '__main__':
    main()