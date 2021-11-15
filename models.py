from itertools import tee
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_roc_curve, roc_curve, roc_auc_score

import matplotlib.pyplot as plt

def make_ds(data, train_split=0.3, shuffle=False):
    X = data.drop(['player0_id', 'player0_name', 'player1_id', 'player1_name', 'winner'], axis=1).to_numpy()
    y = data['winner'].to_numpy()

    return train_test_split(X, y, train_size=train_split, shuffle=shuffle)

def train_models(dataset, models):
    X_train, X_test, y_train, y_test = dataset

    trained = []
    preds = []
    for model in models:
        print(f'Training model -> {model}')
        model.fit(X_train, y_train)
        print('Train acc: %.3f\nTest acc: %.3f\n' % (100*model.score(X_train, y_train), 100*model.score(X_test, y_test)))
        print(classification_report(y_test, model.predict(X_test)))

        trained.append(model)
        preds.append(model.predict(X_test))
        print('='*80)
    return trained, preds


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


    fig = plot_roc_curve(models[0], X_test, y_test)
    for model in models[1:]:
        fig = plot_roc_curve(model, X_test, y_test, ax = fig.ax_)
    plt.show()