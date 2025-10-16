import os
import pywt
import pandas as pd
from numpy import matlib
import random as rn
from CNN_Feat import CNN_Feat
from COA import CO
from Glob_Vars import Glob_Vars
from Model_LSTM import Model_LSTM
from Model_MATCN_AM import Model_MATCN_AM
from Model_SVM import Model_SVM
from Model_S_AM import Model_S_AM
from OOA import OOA
from Objective_function import Objfun_Cls
from POA import POA
from PROPOSED import PROPOSED
from Plot_Results import *
from Spectral_Feature import extract_spectral_features
from Statstical_Feature import extract_statistical_features
from TVETBO import TVETBO

No_of_dataset = 2

# Read Dataset
an = 0
if an == 1:
    path = './Dataset'
    out_dir = os.listdir(path)
    for s in range(len(out_dir)):
        dataset_path = path + '/' + out_dir[s]
        in_dir = os.listdir(dataset_path)
        data = []
        tar = []
        for k in range(len(in_dir)):
            signal_path = dataset_path + '/' + in_dir[k]
            signal = pd.read_csv(signal_path)
            target = signal.iloc[:, -1]
            signal_data = signal.iloc[:, :-1]
            data.append(signal_data)
            tar.append(target)
        np.save('Signal_' + str(s + 1) + '.npy', data)
        np.save('Target_' + str(s + 1) + '.npy', tar)


# Signal Filtering
an = 0
if an == 1:
    for s in range(No_of_dataset):
        Signal = np.load('Signal_' + str(s + 1) + '.npy', allow_pickle=True)
        window_size = 10
        smoothed_signal = np.convolve(Signal, np.ones(window_size) / window_size, mode='same')
        np.save('Denoising_Signal_' + str(s + 1) + '.npy', smoothed_signal)

# Feature Extraction TQWT
an = 0
if an == 1:
    for s in range(No_of_dataset):
        Signal = np.load('Denoising_Signal_' + str(s + 1) + '.npy', allow_pickle=True)
        fs = 1000
        # Perform wavelet decomposition (approximation of TQWT)
        coeffs = pywt.wavedec(Signal, 'db4', level=4)
        # Extract simple features (mean and energy) from each subband
        features = np.array([[np.mean(c), np.sum(c ** 2)] for c in coeffs])
        np.save('Feature_1_' + str(s + 1) + '.npy', features)

# Feature Extraction Deep features from CNN
an = 0
if an == 1:
    for s in range(No_of_dataset):
        Signal = np.load('Denoising_Signal_' + str(s + 1) + '.npy', allow_pickle=True)
        Feature = CNN_Feat(Signal)
        np.save('Feature_2_' + str(s + 1) + '.npy', Feature)

# Feature Extraction Statistical features
an = 0
if an == 1:
    for s in range(No_of_dataset):
        Signal = np.load('Denoising_Signal_' + str(s + 1) + '.npy', allow_pickle=True)
        Feature = extract_statistical_features(Signal)
        np.save('Feature_3_' + str(s + 1) + '.npy', Feature)

# Feature Extraction Spectral Features
an = 0
if an == 1:
    for s in range(No_of_dataset):
        Signal = np.load('Denoising_Signal_' + str(s + 1) + '.npy', allow_pickle=True)
        Feature = extract_spectral_features(Signal)
        np.save('Feature_4_' + str(s + 1) + '.npy', Feature)

# Optimization for Classification
an = 0
if an == 1:
    sol = []
    fitness = []
    for k in range(No_of_dataset):
        Feat_1 = np.load('Feature_1_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feature_2_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_3 = np.load('Feature_3_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_4 = np.load('Feature_4_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(k + 1) + '.npy', allow_pickle=True)
        Glob_Vars.Feat_1 = Feat_1
        Glob_Vars.Feat_2 = Feat_2
        Glob_Vars.Feat_3 = Feat_3
        Glob_Vars.Feat_4 = Feat_4
        Glob_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5, 5, 100], Npop, 1)
        xmax = matlib.repmat([255, 50, 500], Npop, 1)
        fname = Objfun_Cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("CO...")
        [bestfit1, fitness1, bestsol1, time1] = CO(initsol, fname, xmin, xmax, Max_iter)

        print("OOA...")
        [bestfit2, fitness2, bestsol2, time2] = OOA(initsol, fname, xmin, xmax, Max_iter)

        print("POA...")
        [bestfit3, fitness3, bestsol3, time3] = POA(initsol, fname, xmin, xmax, Max_iter)

        print("TVETBO...")
        [bestfit4, fitness4, bestsol4, time4] = TVETBO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        sol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('BestSol.npy', sol)
    np.save('Fitness.npy', fitness)

# Classification
an = 0
if an == 1:
    EV = []
    for k in range(No_of_dataset):
        Feat_1 = np.load('Feature_1_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feature_2_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_3 = np.load('Feature_3_' + str(k + 1) + '.npy', allow_pickle=True)
        Feat_4 = np.load('Feature_4_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(k + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol.npy', allow_pickle=True)[k]
        Feat = np.concatenate((Feat_1, Feat_2, Feat_3, Feat_4), axis=0)
        EVAL = []
        Epoch = [100, 200, 300, 400, 500]
        for learn in range(len(Epoch)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 7))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :] = Model_MATCN_AM(Feat_1, Feat_2, Feat_3, Feat_4, Target, Epoch[learn], sol)
            Eval[5, :] = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epoch[learn])
            Eval[6, :] = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target, Epoch[learn])
            Eval[7, :] = Model_S_AM(Train_Data, Train_Target, Test_Data, Test_Target, Epoch[learn])
            Eval[8, :] = Model_MATCN_AM(Feat_1, Feat_2, Feat_3, Feat_4, Target, Epoch[learn])
            Eval[9, :] = EVAL[4, :]
            EVAL.append(Eval)
        EV.append(EVAL)
    np.save('Eval_all_Epoch.npy', EV)  # Save the Eval_all

Plot_batch_Table()
Plot_Kfold()
plot_roc()
Plot_Batchsize()
plot_results_conv()