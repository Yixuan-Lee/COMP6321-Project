import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_table(df):
    print(df.T)


def plot_bar(df):
    ax = df.plot.bar(rot=0, colormap=plt.cm.Accent, title="compare between balanced data&imbalanced data ")
    ax.set_ylabel("Score")
    ax.set_xlabel("classifier")
    fig = ax.get_figure()
    fig.savefig("bal&im.png")


# def plot_lines(df):
#     df1 = df
#     ax2 = df1.plot(rot=0, colormap=plt.cm.summer, title="Lines Plot")

#     def print_rs(y,y_pred,is_plot = True):
#     analys = np.column_stack((np.zeros((4, 1), dtype=np.float) + sklearn.metrics.accuracy_score(y, y_pred),
#                                   np.array(sklearn.metrics.f1_score(y, y_pred, average=None)).reshape(-1, 1),
#                                   np.array(sklearn.metrics.precision_score(y, y_pred, average=None)).reshape(-1, 1),
#                                   np.array(sklearn.metrics.recall_score(y, y_pred, average=None)).reshape(-1, 1)))
#     index = ["Accuracy", "F1 Score", "Precision Score", "Recall Score"]
#     lables = ["story", "ask_hn", "show_hn", "poll"]
#     if is_plot :
#         plot_confusion_matrix(sklearn.metrics.confusion_matrix(y, y_pred))
#         df = gen_rs(analys,index,lables).T
#         plot_bar(df)
#         plot_table(df)
#     rs = np.sum(analys, axis=0).reshape(-1, 4)/4
#     return rs

def print_rs_list(df):
    df = df.T
    plot_bar(df)
    plot_table(df)
    #plot_lines(df)


def gen_rs(testing_rs, index, lables):
    dic = {}
    for i in range(testing_rs.shape[0]):
        dic[lables[i]] = testing_rs[i, :]

    print(dic)
    df = pd.DataFrame(dic, index=index, dtype=np.float)
    print(df)
    return df


testing_rs = np.array([[76, 45.79, 61.45],
                       [82.61, 68.13, 36.35],
                       [76.11, 45.96, 61.31],
                       [82.48, 69.19, 33.86],
                       [61.19, 31.98, 70.96],
                       [81.65, 73.95, 23],
                       [81.06, 63.35, 28.90],
                       [80.9, 61.09, 31.48]
                       ]).transpose()
index = ["DT(bal)","DT(im)" ,"SVC(bal)","SVC(im)", "lg(bal)",  "lg(im)",\
         "rf(bal)","rf(im)"]
lables = ["accuracy", "precision", "recall"]

df = gen_rs(testing_rs, index, lables)

plot_bar(df)
# plot_table(df)
# df.plot(rot=0, colormap=plt.cm.summer, title="Lines Plot")
plt.show()