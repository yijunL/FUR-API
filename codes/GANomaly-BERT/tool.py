import pandas as pd


def data_cut():

    data = pd.read_csv('./data/train.csv')

    validation_data = data.head(2007)
    print(validation_data)

    training_data = data[2007:]


    validation_data.to_csv('./data/dev.csv', index=False)

    training_data.to_csv('./data/training.csv', index=False)


def data_separate(name):
    import pandas as pd


    input_file_path = "./data/" + name + ".csv"
    data = pd.read_csv(input_file_path)


    normal_data = data[data["type"] == "normal"]
    abnormal_data = data[data["type"] == "anomaly"]

    normal_output_path = "./data/" + name + "_n.csv"
    abnormal_output_path = "./data/" + name + "_a.csv"

    normal_data.to_csv(normal_output_path, index=False)
    abnormal_data.to_csv(abnormal_output_path, index=False)



data_separate("train")
data_separate("dev")
data_separate("test")
