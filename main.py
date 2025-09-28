import os

os.makedirs('augmented_images', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)
os.makedirs('Data Visualization', exist_ok=True)

from Augmentation import Augmentation
from datagen import datagen
from save_load import load, save
# from Classification import proposed, alexnet, Resnet, cnn, inception_v3, san_cnn, efficientnet_b0
# from Fitness_Function import fit_func_70, fit_func_80
from WSGPOA import WSGPOA
import matplotlib.pyplot as plt
from plot_result import plotres


def main():
    # Data Augmentation
    Augmentation()

    # Data Preprocessing and Feature Extraction
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70, fit_func_70),
                     (x_train_80, y_train_80, x_test_80, y_test_80, fit_func_80)]

    i = 70

    for train_data in training_data:
        X_train, y_train, X_test, y_test, fit_func = train_data

        # Parameter Tuning
        lb = [50, 16, 0.00001]  # epochs, batch_size, learning-rate
        ub = [100, 32, 0.001]

        pop_size = 10
        prob_size = len(lb)

        max_iter = 10
        best_solution, best_fitness = WSGPOA(fit_func, prob_size, pop_size, max_iter, lb, ub)

        epochs = int(best_solution[0])  # optimal epoch for 70 is 83 and 80 is 95
        learning_rate = best_solution[2]  # optimal learning rate for 70 is 0.001 and 80 is 0.0001
        batch_size = int(best_solution[1])  # optimal batch_size for 70 is 32 abd 80 is 18

        y_pred, met, history = proposed(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate)

        save(f'proposed_{i}', met)
        save(f'predicted_{i}', y_pred)

        plt.figure(figsize=(10, 4))

        plt.subplot(121)
        plt.plot(history.history['accuracy'], label=['Train Accuracy'])
        plt.plot(history.history['val_accuracy'], label=['Validation Accuracy'])
        plt.title('Accuracy', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('Accuracy', fontweight='bold', fontname='Serif')
        plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
        plt.yticks(fontweight='bold', fontname='Serif')
        plt.legend(loc='lower right', prop={'weight': 'bold', 'family': 'Serif'})

        plt.subplot(122)
        plt.plot(history.history['loss'], label=['Train Loss'])
        plt.plot(history.history['val_loss'], label=['Validation Loss'])
        plt.title('Loss', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('Loss', fontweight='bold', fontname='Serif')
        plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
        plt.yticks(fontweight='bold', fontname='Serif')
        plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})

        plt.tight_layout()
        plt.savefig(f'Results/Accuracy Loss Graph Learning rate {i}.png')
        plt.show()

        pred, met = cnn(X_train, y_train, X_test, y_test)
        save(f'cnn_{i}', met)

        pred, met = alexnet(X_train, y_train, X_test, y_test)
        save(f'alexnet_{i}', met)

        pred, met = Resnet(X_train, y_train, X_test, y_test)
        save(f'resnet_{i}', met)

        pred, met = inception_v3(X_train, y_train, X_test, y_test)
        save(f'inception_v3_{i}', met)

        pred, met = san_cnn(X_train, y_train, X_test, y_test)
        save(f'san_cnn_{i}', met)

        pred, met = efficientnet_b0(X_train, y_train, X_test, y_test)
        save(f'efficient_net_{i}', met)



        i += 10


a = 0
if a == 1:
    main()

plotres()
plt.show()







