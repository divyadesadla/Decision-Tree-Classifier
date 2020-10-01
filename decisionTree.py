import sys
import numpy as np
import csv
import math

# Defining a class


class MyDecisionTree:
    def __init__(self, left=None, right=None, attribute=None, attribute_value=None, attribute_outputs=None):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.attribute_outputs = attribute_outputs

# Entropy function


def get_entropy(Data):
    Total = len(Data)
    entropy = 0
    labels = dict()
    for i in range(Total):
        label = Data[i]
        labels[label] = labels.get(label, 0) + 1

    for i in labels.keys():
        probability = labels[i]/Total
        if probability != 0:
            logvalue = math.log2(probability)
            entropy += (probability * logvalue * (-1))
    return entropy


def decisiontree(data, max_depth, attributenames, attributevalue, attribute, current_depth=0):
    Total = len(data)
    num_col = len(data[0]) - 1
    entropy = get_entropy(data[:, -1])
    infogain = []
    maxinfogain = 0

    for k in range(num_col):
        binaryattribute = np.unique(data[:, k])

        type1 = []
        type2 = []
        # For each attribute type, getting output counts
        for i in range(Total):
            if data[i, k] == binaryattribute[0]:
                type1.append(data[i, -1])
            else:
                type2.append(data[i, -1])

        prob1 = len(type1)/Total
        prob2 = len(type2)/Total
        print(prob1)
        print(prob2)

        # Calculating Mutual Information
        if prob1 > 0:
            cond_entropy1 = get_entropy(type1)*prob1
        else:
            cond_entropy1 = 0

        if prob2 > 0:
            cond_entropy2 = get_entropy(type2)*prob2
        else:
            cond_entropy2 = 0
        print(cond_entropy1)
        print(cond_entropy2)
        temp = entropy - cond_entropy1 - cond_entropy2
        infogain.append(temp)
    print(infogain)
    # Calculating Maximum Mutual Information
    maxinfogain = np.max(infogain)
    # print(maxinfogain)

    # To check which coloumn has the max MI
    maxinfogain_index = np.argmax(infogain)

    type1 = []
    type2 = []
    binaryattribute = np.unique(data[:, maxinfogain_index])

    # Splitting attribute types in column with max MI.
    for i in range(Total):
        if data[i, maxinfogain_index] == binaryattribute[0]:
            type1.append(data[i])
        else:
            type2.append(data[i])

    # Getting the labels for that node
    labels = dict()
    for i in range(Total):
        label = data[i, -1]
        labels[label] = labels.get(label, 0) + 1

    type1 = np.asarray(type1)
    type2 = np.asarray(type2)

    # Creating left and right child for nodes
    if maxinfogain > 0 and current_depth < max_depth:
        leftnode = decisiontree(type1, max_depth, attributenames,
                                binaryattribute[0], attributenames[maxinfogain_index], current_depth+1)
        rightnode = decisiontree(type2, max_depth, attributenames,
                                 binaryattribute[1], attributenames[maxinfogain_index], current_depth+1)
        return MyDecisionTree(leftnode, rightnode, attribute, attributevalue, labels)
    else:
        return MyDecisionTree(None, None, attribute, attributevalue, labels)


def printTree(tree, indent=''):
    if tree.left != None and tree.right != None:
        print(indent, tree.attribute, "=",
              tree.attribute_value + ":", tree.attribute_outputs)
        printTree(tree.left, indent+'| ')
        printTree(tree.right, indent+'| ')
    else:
        print(indent, tree.attribute, "=",
              tree.attribute_value + ":", tree.attribute_outputs)


def tree_traversal(tree, data, names):
    Total = len(data)
    out = []
    for i in range(Total):
        prediction = check_output(tree, data[i], names)
        out.append(prediction)

    return out


def check_output(tree, row, names):
    while (tree.left != None and tree.right != None):
        attribute = tree.left.attribute
        index = 0
        for i in range(len(names)):
            if attribute == names[i]:
                index = i
                break

        value = row[index]

        if value == tree.left.attribute_value:
            tree = tree.left
        else:
            tree = tree.right

    attribute_outputs = tree.attribute_outputs
    Majority = max(attribute_outputs.keys(), key=(
        lambda k: attribute_outputs[k]))
    # print(Majority)
    return Majority


def compare_out(og_out, predict_out):
    Total = len(og_out)
    count = 0
    for i in range(Total):
        if og_out[i] != predict_out[i]:
            count = count + 1

    error = count/Total

    return error


if __name__ == '__main__':
    metrics_out = sys.argv[-1]
    arg_test_out = sys.argv[-2]
    arg_train_out = sys.argv[-3]
    max_depth = sys.argv[-4]
    test_input = sys.argv[-5]
    train_input = sys.argv[-6]

    traindatain = open(train_input, 'r')
    testdatain = open(test_input, 'r')
    # data = np.genfromtxt(traindatain, dtype='str', delimiter=',')
    # testdata = np.genfromtxt(testdatain, dtype='str', delimiter=',')
    data_csv = csv.reader(traindatain, delimiter=',')
    data_test_csv = csv.reader(testdatain, delimiter=',')

    data = list()
    for i in data_csv:
        data.append(i)

    data = np.asarray(data)

    testdata = list()
    for i in data_test_csv:
        testdata.append(i)

    testdata = np.asarray(testdata)

    attributenames = data[0]

    decisionTree = decisiontree(data[1:, :], int(
        max_depth), attributenames, '', '')
    # printTree(decisionTree)

    train_out = tree_traversal(decisionTree, data[1:, :], attributenames)

    test_out = tree_traversal(decisionTree, testdata[1:, :], attributenames)

    train_error = compare_out(data[1:, -1], train_out)
    test_error = compare_out(testdata[1:, -1], test_out)

    file6 = open(metrics_out, 'w')
    ans1 = 'error(train): '+str(train_error)
    ans2 = 'error(test): '+str(test_error)
    file6.write(ans1+'\n')
    file6.write(ans2)
    file6.close()

    file5 = open(arg_test_out, 'w')
    for i in test_out:
        file5.write(str(i)+'\n')
    file5.close()

    file4 = open(arg_train_out, 'w')
    for i in train_out:
        file4.write(str(i)+'\n')
    file4.close()
