import pandas as pd


def data_cut():
    # 读取原始数据集
    data = pd.read_csv('./data/train.csv')

    # 抽取前 10 条数据作为验证集
    validation_data = data.head(2007)
    print(validation_data)
    # 剩余的数据作为训练集
    training_data = data[2007:]

    # 将验证集保存到一个新的 CSV 文件
    validation_data.to_csv('./data/dev.csv', index=False)

    # 将训练集保存到一个新的 CSV 文件
    training_data.to_csv('./data/training.csv', index=False)


def data_separate(name):
    import pandas as pd

    # 读取原始CSV文件
    input_file_path = "./data/" + name + ".csv"
    data = pd.read_csv(input_file_path)

    # 根据"type"字段将数据分成正常和异常两部分
    normal_data = data[data["type"] == "normal"]
    abnormal_data = data[data["type"] == "anomaly"]

    # 将分开的数据保存到两个不同的CSV文件中
    normal_output_path = "./data/" + name + "_n.csv"
    abnormal_output_path = "./data/" + name + "_a.csv"

    normal_data.to_csv(normal_output_path, index=False)
    abnormal_data.to_csv(abnormal_output_path, index=False)

    print(name + "数据已根据类型分开并保存到CSV文件中。")


data_separate("train")
data_separate("dev")
data_separate("test")