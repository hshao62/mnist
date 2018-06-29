THRESHOLD = 0.5


def buildConfusionMatrix(p, labels, numLabels):
    if numLabels == 1:
        return build2DConfusionMatrix(p, labels)
    else:
        return buildNDConfusionMatrix(p, labels, numLabels)


def build2DConfusionMatrix(p, labels):
    preds = []
    for i in range(len(p)):
        if p[i][0] > THRESHOLD:
            preds.append(1)
        else:
            preds.append(0)

    matrix = {}
    for i in range(len(preds)):
        actual = labels[i][0]
        predicted = preds[i]

        if actual not in matrix:
            matrix[actual] = {0: 0, 1: 0}

        if predicted not in matrix[actual]:
            matrix[actual][predicted] = 1
        else:
            matrix[actual][predicted] += 1

    return matrix


def buildNDConfusionMatrix(p, labels, numLabels):
    matrix = {}
    for i in range(numLabels):
        matrix[i] = {}
        for j in range(numLabels):
            matrix[i][j] = 0

    for i in range(len(p)):
        actual = 0
        predicted = 0

        for j in range(1, len(p[i])):
            if labels[i][j] > labels[i][actual]:
                actual = j

            if p[i][j] > p[i][predicted]:
                predicted = j

        if predicted not in matrix[actual]:
            matrix[actual][predicted] = 1
        else:
            matrix[actual][predicted] += 1

    return matrix


def printConfusionMatrix(matrix, possibleLabels):
    rows = list(matrix.keys())
    cols = list(matrix.keys())
    rows.sort()
    cols.sort()

    # print the column header
    header = ""
    for col in cols:
        header += str(possibleLabels[col]) + "\t"
    print(header)

    # print each row
    for row in rows:
        for col in cols:
            print("%03d " % matrix[row][col], end="\t")
        print("|", possibleLabels[row])


def printAccuracy(matrix):
    total = 0
    correct = 0
    for i in matrix:
        for j in matrix[i]:
            total += matrix[i][j]
            if i == j:
                correct += matrix[i][j]

    #print("Accuracy:", "{0:.4f}".format(correct / total))
    return "{0:.4f}".format(correct/total)