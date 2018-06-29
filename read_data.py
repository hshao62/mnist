import random

def read_data(filename, train_perc, label_index, num_labels):
    with open(filename, 'r', newline='') as data:
        columns = data.readline().split(',')
        rows = []

        for line in data:
            row = line.split(',')
            for i in range(0, len(row)):
                row[i] = row[i].rstrip()

            rows.append(row)

    random.shuffle(rows)

    attributes = []
    labels = []

    for ele in rows:
        attributes.append(ele[:label_index] + ele[label_index+1:])
        labels.append(ele[label_index])

    train_index = int(len(attributes) * train_perc // 1)
    label_index = {}
    index = 0
    for i in range(len(labels)):
        if labels[i] in label_index:
            new = [0] * num_labels
            new[label_index[labels[i]]] = 1
            labels[i] = new
        else:
            label_index[labels[i]] = index
            new = [0] * num_labels
            new[label_index[labels[i]]] = 1
            labels[i] = new
            index += 1

    possible_labels = []
    for key in label_index.keys():
        possible_labels.append(key)
    train_x = attributes[:train_index]
    train_y = labels[:train_index]

    test_x = attributes[train_index:]
    test_y = labels[train_index:]

    return (train_x, train_y), (test_x, test_y), possible_labels
