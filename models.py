import numpy as np
from itertools import tee
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_roc_curve, roc_curve, roc_auc_score

import matplotlib.pyplot as plt

def make_ds(data, train_split=0.7, shuffle=False):
    X = data.drop(['player0_rank', 'player1_rank', 'player0_id', 'player0_name', 'player1_id', 'player1_name', 'round', 'minutes', 'winner'], axis=1).to_numpy()
    y = data['winner'].to_numpy()

    if train_split==1:
        return [X, np.array([]), y, np.array([])]
    if train_split==0:
        return [np.array([]), X, np.array([]), y]

    return train_test_split(X, y, train_size=train_split, shuffle=shuffle)

def train_models(dataset, models):
    X_train, X_test, y_train, y_test = dataset

    trained = []
    for model in models:
        print(f'Training model -> {model}')
        model.fit(X_train, y_train)
        print('Train acc: %.3f\nTest acc: %.3f\n' % (100*model.score(X_train, y_train), 100*model.score(X_test, y_test)))
        print(classification_report(y_test, model.predict(X_test)))

        trained.append(model)
        print('='*80)
    return trained

def get_preds(dataset, models):
    X_train, X_test, y_train, y_test = dataset

    preds = []
    for model in models:
        preds.append(model.predict(X_test))
    return preds

def get_probas(dataset, models):
    X_train, X_test, y_train, y_test = dataset

    probas = []
    for model in models:
        probas.append(model.predict_proba(X_test).max(axis=1))
    return np.array(probas)

def hard_voting(preds):
    preds = np.array(preds)
    return (preds.sum(axis=0)>len(preds)/2)*1

def soft_voting(probas):
    preds = np.array(probas)
    return (preds.mean(axis=0)>0.5)*1

def plot_rocs(dataset, models, preds):
    X_train, X_test, y_train, y_test = dataset

    # plt.figure()
    # for i in range(len(models)):
    #     fpr, tpr, _ = roc_curve(y_test, preds[i])
    #     auc = roc_auc_score(y_test, preds[i])
    #     plt.plot(fpr,tpr,label='%s, auc=%.3f' % (models[i].__str__().split('(')[0], auc))
    
    # plt.xlabel('1-Specificity(False Positive Rate)')
    # plt.ylabel('Sensitivity(True Positive Rate)')
    # plt.title('ROC Curves')
    # plt.legend(loc="lower right")
    # plt.show()

    plt.figure()
    fig = plot_roc_curve(models[0], X_test, y_test)
    for model in models[1:]:
        fig = plot_roc_curve(model, X_test, y_test, ax = fig.ax_)
    plt.show()
    return fig

def plot_bar(title, models, acc, labels, nn_acc=None):
    if nn_acc:
        plt.bar(
            [model.__str__().split('(')[0] for model in models]+labels,
            np.hstack([acc, nn_acc])
        )
    else:
        plt.bar(
            [model.__str__().split('(')[0] for model in models],
            acc
        )
    plt.xticks(rotation='vertical')
    plt.title(title)
    plt.show()