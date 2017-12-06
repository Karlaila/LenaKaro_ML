from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree


def parse(path_feature='train_data.csv', path_labels='train_labels.csv', data_limit = -1):
    X = []
    with open(path_feature, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in reader:
            if counter > 0:
                X.append([float(i) for i in row])
            if data_limit > 0:
                if counter > data_limit:
                    break
            counter+=1

    counter = 0
    Y = []
    with open(path_labels, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if counter > 0:
                Y.append(float(row[0]))
            if data_limit > 0:
                if counter > data_limit:
                    break
            counter+=1

    return X, Y


def divide_set(features, labels):
    X_test, Y_test, X_train, Y_train = [], [], [], []

    # dividing into test and train data
    for i in range(0, len(labels)):
        if i % 4 == 0:
            X_test.append(features[i][1:265])
            Y_test.append(labels[i])
        else:
            X_train.append(features[i][1:265])
            Y_train.append(labels[i])
    return X_test, Y_test, X_train, Y_train


def check (Y_test, Y_pred):
    ok = 0
    notok = 0
    for i in range(0, len(Y_test)):
        if Y_pred[i] == Y_test[i]:
            ok += 1
        else:
            notok += 1
            # if i < 900:
            #    print result, Y_test[i]
    return "ok: ", ok, "; not ok:", notok, "; percentage: ", (ok * 100) / (notok + ok)

# for followitg features the best score
def do_svc(X_test, Y_test, X_train, Y_train):
    clf = svm.SVC(decision_function_shape='ovo', class_weight='balanced', kernel = "poly")
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = []
    for i in range(0, len(Y_test)):
        Y_pred.append(clf.predict([X_test[i]]))
    return Y_pred


# very random, no matter of loss and penalty functions, between 35-56 percent
def do_sgd(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = SGDClassifier(loss="hinge", penalty="l2")
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = []
    for i in range(0, len(Y_test)):
        Y_pred.append(clf.predict([X_test[i]]))
    return Y_pred


# very low score (around 30)
def do_dec_tree(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = tree.DecisionTreeClassifier()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = []
    for i in range(0, len(Y_test)):
        Y_pred.append(clf.predict([X_test[i]]))
    return Y_pred


# ?
def do_gpc(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = tree.DecisionTreeClassifier()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = []
    for i in range(0, len(Y_test)):
        Y_pred.append(clf.predict([X_test[i]]))
    return Y_pred
