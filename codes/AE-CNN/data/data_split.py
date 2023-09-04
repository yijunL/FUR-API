import csv
import random

with open('train.csv', 'r', errors='ignore') as file:

    csv_reader = csv.reader(file)
    data = []

    for row in csv_reader:
        data.append(row)
        
header = [data[0]]
data = data[1:]

random.shuffle(data)
nor = []
ano = []
for i in data:
    if(i[-1]=="normal"):
        nor.append(i)
    else:
        ano.append(i)

with open('train_n.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
    writer.writerows(nor[:int(0.9*len(nor))])

with open('train_a.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
    writer.writerows(ano)

with open('dev_n.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
    writer.writerows(nor[int(0.9*len(nor)):])
