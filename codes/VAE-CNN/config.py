# coding: UTF-8
import os
import torch


class Config(object):
    def __init__(self, data_dir):
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "train_n.csv")
        self.dev_file = os.path.join(data_dir, "dev_n.csv")
        self.test_file = os.path.join(data_dir, "test.csv")
        self.train_anomaly_file = os.path.join(data_dir, "train_a.csv")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)
        assert os.path.isfile(self.test_file)

        self.saved_model_dir = os.path.join(data_dir, "model")
        self.saved_model = os.path.join(self.saved_model_dir, "cnn_ae_model.pth")
        if not os.path.exists(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)

        self.label_list = ["normal", "abnormal"]
        self.num_labels = 2

        self.num_epochs = 10
        self.log_batch = 200
        self.batch_size = 16
        self.max_seq_len = 512
        self.require_improvement = 1000

        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
