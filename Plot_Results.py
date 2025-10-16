import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import cv2 as cv


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_roc():
    lw = 2
    cls = ['LSTM', 'SVM', 'S-AM', 'MATCN-AM', 'ETVETBO-MATCN-AM']
    colors = cycle(["#5a86ad", "#84597e", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):
        Dataset = ['Dataset 1', 'Dataset 2']
        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontweight='bold', fontsize=14)
        plt.ylabel("True Positive Rate", fontweight='bold', fontsize=14)
        plt.title("ROC Curve", fontweight='bold', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        path = "./Results/%s_Roc_.png" % (Dataset[i])
        plt.savefig(path, dpi=600)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def Plot_batch_Table():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Terms = [0, 1, 2, 3, 4, 8, 9]
    Algorithm = ['TERMS', 'CO-MATCN-AM', 'OOA-MATCN-AM', 'POA-MATCN-AM', 'TVETBO-MATCN-AM', 'ETVETBO-MATCN-AM']
    Classifier = ['TERMS', 'LSTM', 'SVM', 'S-AM', 'MATCN-AM', 'ETVETBO-MATCN-AM']
    Activation = ['1', '2', '3', '4', '5']
    Dataset = ['Dataset 1', 'Dataset 2']

    for i in range(2):
        value = eval[i, 4, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Terms])
        print('-------------------------------------------------- ',Dataset[i], 'K Fold ',
              'Algorithm Comparison --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Terms])
        print('-------------------------------------------------- ',Dataset[i], ' K Fold ',
              'Classifier Comparison --------------------------------------------------')
        print(Table)


def Plot_Kfold():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 1, 2, 3, 4]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='*', markerfacecolor='red',
                    markersize=12,
                    label='CO-MATCN-AM')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='*', markerfacecolor='green',
                    markersize=12,
                    label='OOA-MATCN-AM')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='*', markerfacecolor='cyan',
                    markersize=12,
                    label='POA-MATCN-AM')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='*', markerfacecolor='#fdff38',
                    markersize=12,
                    label='TVETBO-MATCN-AM')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='k', markersize=12,
                    label='ETVETBO-MATCN-AM')
            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('K - Fold', fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Dataset_%s_kfold_%s_Alg.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path, dpi=600)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='//', color='#94568c', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='-', color='#8f99fb', width=0.10, label="SVM")
            ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='//', color='#eecffe', width=0.10, label="S-AM")
            ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='-', color='lime', width=0.10, label="MATCN-AM")
            ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='..', color='k', width=0.10, label="ETVETBO-MATCN-AM")
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('K - Fold', fontweight='bold', fontsize=14)
            plt.ylabel(Terms[Graph_Term[j]], fontweight='bold', fontsize=14)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_kfold_%s_Med.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path, dpi=600)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def Plot_Batchsize():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 2, 3, 4, 9, 12, 16]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                    markersize=12,
                    label='CO-MATCN-AM')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='>', markerfacecolor='#017b92',
                    markersize=12,
                    label='OOA-MATCN-AM')
            ax.plot(length, Graph[:, 2], color='#eecffe', linewidth=3, marker='>', markerfacecolor='cyan',
                    markersize=12,
                    label='POA-MATCN-AM')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#eecffe',
                    markersize=12,
                    label='TVETBO-MATCN-AM')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='>', markerfacecolor='k', markersize=12,
                    label='ETVETBO-MATCN-AM')

            ax.fill_between(length, Graph[:, 0], Graph[:, 3], color='#017b92', alpha=.5)  # ff8400
            ax.fill_between(length, Graph[:, 3], Graph[:, 2], color='#658cbb', alpha=.5)  # 19abff
            ax.fill_between(length, Graph[:, 2], Graph[:, 1], color='#3b5b92', alpha=.5)  # 00f7ff
            ax.fill_between(length, Graph[:, 1], Graph[:, 4], color='#b2fba5', alpha=.5)  # ecfc5b
            plt.xticks(length, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Dataset_%s_Epoch_%s_Alg.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path, dpi=600)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='//', color='r', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='-', color='#fe46a5', width=0.10, label="SVM")
            ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='//', color='lime', width=0.10, label="S-AM")
            ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='-', color='#9e43a2', width=0.10, label="MATCN-AM")
            ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='..', color='k', width=0.10, label="ETVETBO-MATCN-AM")
            plt.xticks(X + 0.25, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs', fontweight='bold', fontsize=14)
            plt.ylabel(Terms[Graph_Term[j]], fontweight='bold', fontsize=14)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Epoch_%s_Med.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path, dpi=600)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CO-MATCN-AM', 'OOA-MATCN-AM', 'POA-MATCN-AM', 'TVETBO-MATCN-AM', 'ETVETBO-MATCN-AM']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Dataset = ['Dataset1', 'Dataset2']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ',Dataset[i],  'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='CO-MATCN-AM')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='.', markerfacecolor='#fe83cc',
                 markersize=12,
                 label='OOA-MATCN-AM')
        plt.plot(length, Conv_Graph[2, :], color='#9e43a2', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='POA-MATCN-AM')
        plt.plot(length, Conv_Graph[3, :], color='#fe83cc', linewidth=3, marker='.', markerfacecolor='magenta',
                 markersize=12,
                 label='TVETBO-MATCN-AM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='#9e43a2',
                 markersize=12,
                 label='ETVETBO-MATCN-AM')
        plt.xlabel('Iteration', fontweight='bold', fontsize=14)
        plt.ylabel('Cost Function', fontweight='bold', fontsize=14)
        plt.legend(loc=1, fontsize=12)
        plt.savefig("./Results/Conver_%s.png" % (Dataset[i]), dpi=600)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    Plot_batch_Table()
    Plot_Kfold()
    plot_roc()
    Plot_Batchsize()
    plot_results_conv()
