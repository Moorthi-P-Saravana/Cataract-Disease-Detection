import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from save_load import load
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import joblib


def classfi_report(y_test, predicted, k):

    # Classification report
    class_report = classification_report(y_test, predicted, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()

    # Plot the DataFrame
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')

    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.title('Classification Report', fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Classification Report learning rate - {k}.png')
    plt.show()


def line_plot(label, data1, data2, metric):
    df = pd.DataFrame({'Learn Rate - 70': data1, 'Learn Rate - 80': data2}, index=label)

    plt.figure(figsize=(8, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)

    # Customizing plot
    plt.ylabel(metric, fontweight='bold', fontname='Serif')
    plt.xlabel('Learn Rate', fontweight='bold', fontname='Serif')
    plt.xticks(fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.legend(loc='best', prop={'weight': 'bold', 'family': 'Serif', 'size': 9})
    plt.title(f'{metric}', fontweight='bold', fontname='Serif')

    # Save and show
    plt.savefig(f'./Results/{metric}.png', dpi=400)
    plt.show(block=False)


def taylor_diagram(actual, predicted, learning_rate):
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Compute statistics
    std_actual = np.std(actual)
    std_predicted = np.std(predicted)
    correlation = np.corrcoef(actual, predicted)[0, 1]
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Polar plot setup
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)

    # Convert correlation to angle
    theta = np.arccos(correlation)

    # Plot standard deviations as circles
    std_range = np.linspace(0, max(std_actual, std_predicted) * 1.5, 100)
    for std in std_range[::20]:
        ax.plot(np.linspace(0, np.pi / 2, 50), [std] * 50, 'k--', alpha=0.3)

    # Plot RMSE contours
    for r in np.linspace(0, max(std_actual, std_predicted) * 1.5, 4)[1:]:
        ax.plot([0, np.pi / 2], [r, r], 'g--', alpha=0.3)

    # Plot reference point (Actual Data)
    ax.scatter(0, std_actual, color='red', label='Actual', s=100)

    # Plot predicted point
    ax.scatter(theta, std_predicted, color='blue', label='Predicted', s=100)

    # Labels and legend
    ax.set_xlabel('Correlation (Cosine of Angle)', fontweight='bold', fontname='Serif')
    ax.set_ylabel('Standard Deviation', fontweight='bold', fontname='Serif')
    ax.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})
    ax.set_title(f"Taylor Diagram (LR-{learning_rate})", fontweight='bold', fontname='Serif')

    plt.savefig(f'Results/Taylor_Diagram_LR-{learning_rate}.png')
    plt.show()


def confu_plot(y_test, y_pred, k):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Purples, values_format='.0f', ax=ax)

    plt.ylabel('True Labels', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Predicted Labels', fontweight='bold', fontname='Serif')
    plt.title('Confusion Matrix', fontweight='bold', fontname='Serif')
    plt.tight_layout()
    plt.savefig(f'Results/Confusion Matrix Learning rate-{k}.png')
    plt.show()


def precision_recall_plot(y_test, y_pred, k):
    # Binarize the output labels
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    # Compute Precision-Recall curve and plot
    precision = dict()
    recall = dict()
    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(8, 6))
    colors = cycle(
        ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'purple', 'brown', 'pink'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'.format(i, auc(recall[i], precision[i])))

    plt.ylabel('Precision', fontweight='bold', fontname='Serif')
    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Recall', fontweight='bold', fontname='Serif')
    plt.legend(loc="lower left", prop={'weight': 'bold', 'family': 'Serif'})
    plt.title('Precision-Recall curve for multi-class data', fontweight='bold', fontname='Serif')
    plt.tight_layout()
    plt.savefig(f'Results/Precision Recall Curve - learning rate - {k}.png')
    plt.show()


def plotres():
    # learning rate -  70  and 30

    cnn_70 = load('cnn_70')
    alexnet_70 = load('alexnet_70')
    resnet_70 = load('resnet_70')
    inception_v3_70 = load('inception_v3_70')
    proposed_70 = load('proposed_70')

    data = {
        'CNN': cnn_70,
        'AlexNet': alexnet_70,
        'ResNet': resnet_70,
        'Inception V3': inception_v3_70,
        'PROPOSED': proposed_70
    }

    ind = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC',
           'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table = pd.DataFrame(data, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)

    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # learning rate -  80  and 20

    cnn_80 = load('cnn_80')
    alexnet_80 = load('alexnet_80')
    resnet_80 = load('resnet_80')
    inception_v3_80 = load('inception_v3_80')
    proposed_80 = load('proposed_80')

    data1 = {
        'CNN': cnn_80,
        'AlexNet': alexnet_80,
        'ResNet': resnet_80,
        'Inception V3': inception_v3_80,
        'PROPOSED': proposed_80
    }

    ind = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC',
           'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table1 = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)

    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['CNN', 'AlexNet', 'ResNet', 'Inception V3', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score',
                     'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']

    # Line plots
    for i in range(len(metrices_plot)):
        plt.figure(figsize=(8, 6))
        plt.plot(mthod, metrices[0][i, :], marker='o', label='LR-70')
        plt.plot(mthod, metrices[1][i, :], marker='s', label='LR-80')

        plt.xlabel('Models', fontweight='bold', fontname='Times New Roman', fontsize=13)
        plt.ylabel(metrices_plot[i], fontweight='bold', fontname='Times New Roman', fontsize=13)
        plt.xticks(fontweight='bold', fontname='Times New Roman', fontsize=13)
        plt.yticks(fontweight='bold', fontname='Times New Roman', fontsize=13)
        plt.title(metrices_plot[i], fontweight='bold', fontname='Times New Roman', fontsize=15)
        plt.legend(prop={'weight': 'bold', 'family': 'Times New Roman', 'size': 13})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./Results/{metrices_plot[i]}.png', dpi=400)
        plt.show()

    learn_data = [70, 80]
    for k in learn_data:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')

        label_encoder = joblib.load('Saved Data/label encoder.joblib')

        taylor_diagram(y_test, y_pred, k)

        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)

        classfi_report(y_test, y_pred, k)
        confu_plot(y_test, y_pred, k)
        precision_recall_plot(y_test, y_pred, k)

    df = pd.read_excel("k fold metrics.xlsx")

    # Extract metrics and relevant columns
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall']
    df_plot = df[df['Metric'].isin(metrics_to_plot)]

    metrics = df_plot['Metric'].values
    k_folds = ['K1', 'K2', 'K3', 'K4', 'K5']
    bar_width = 0.15
    x = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))
    for i, fold in enumerate(k_folds):
        values = df_plot[fold].values
        plt.bar(x + i * bar_width, values, width=bar_width, label=fold)

    plt.xlabel("Metrics", fontweight='bold', fontname='Times New Roman', fontsize=13)
    plt.ylabel("Scores", fontweight='bold', fontname='Times New Roman', fontsize=13)
    plt.title("Accuracy, Precision, Recall across K-Folds",
              fontweight='bold', fontname='Times New Roman', fontsize=15)
    plt.xticks(x + bar_width * 2, metrics, fontweight='bold',
               fontname='Times New Roman', fontsize=13)
    plt.yticks(fontweight='bold', fontname='Times New Roman', fontsize=13)
    plt.ylim(0.95, 1.0)
    plt.legend(title="Folds", prop={'weight': 'bold', 'family': 'Times New Roman', 'size': 13})

    plt.tight_layout()
    plt.savefig('Results/k fold cross validation.png', dpi=400)
    plt.show()


    comp = pd.read_excel('Results/comparison.xlsx')
    print(comp.columns)
    print(comp)

    font_config = {'fontweight': 'bold', 'fontname': 'Serif'}

    # Graph 1: Accuracy, Precision, Recall
    plt.figure(figsize=(10, 6))
    for metric in ['Accuracy', 'Precision', 'Recall']:
        values = comp[comp['Metric'] == metric].iloc[0, 1:]
        plt.plot(values.index, values.values, label=metric, linewidth=2, marker='o')

    plt.title('Accuracy, Precision, Recall', **font_config)
    plt.xlabel('Models', **font_config)
    plt.ylabel('Score', **font_config)
    plt.xticks(**font_config)
    plt.yticks(**font_config)
    plt.legend(prop={'weight': 'bold', 'family': 'Serif'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graph1_accuracy_precision_recall.png')
    plt.show()

    # Graph 2: F1 Score, R2 Score, MCC
    plt.figure(figsize=(10, 6))
    for metric in ['F1 Score', 'R2 Score', 'MCC']:
        values = comp[comp['Metric'] == metric].iloc[0, 1:]
        plt.plot(values.index, values.values, label=metric, linewidth=2, marker='o')

    plt.title('F1 Score, R2 Score, MCC', **font_config)
    plt.xlabel('Models', **font_config)
    plt.ylabel('Score', **font_config)
    plt.xticks(**font_config)
    plt.yticks(**font_config)
    plt.legend(prop={'weight': 'bold', 'family': 'Serif'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graph2_f1_r2_mcc.png')
    plt.show()

    # Graph 3: Cohen’s Kappa, Jaccard Score
    plt.figure(figsize=(10, 6))
    for metric in ['Cohen’s Kappa', 'Jaccard Score']:
        values = comp[comp['Metric'] == metric].iloc[0, 1:]
        plt.plot(values.index, values.values, label=metric, linewidth=2, marker='o')

    plt.title('Cohen’s Kappa & Jaccard Score', **font_config)
    plt.xlabel('Models', **font_config)
    plt.ylabel('Score', **font_config)
    plt.xticks(**font_config)
    plt.yticks(**font_config)
    plt.legend(prop={'weight': 'bold', 'family': 'Serif'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graph3_kappa_jaccard.png')
    plt.show()

    # Graph 4: Hamming Loss
    plt.figure(figsize=(10, 6))
    metric = 'Hamming Loss'
    values = comp[comp['Metric'] == metric].iloc[0, 1:]
    plt.plot(values.index, values.values, marker='o', color='red', label=metric, linewidth=2)

    plt.title('Hamming Loss', **font_config)
    plt.xlabel('Models', **font_config)
    plt.ylabel('Loss', **font_config)
    plt.xticks(**font_config)
    plt.yticks(**font_config)
    plt.legend(prop={'weight': 'bold', 'family': 'Times New Roman'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graph4_hamming_loss.png')
    plt.show()
    x = 0


