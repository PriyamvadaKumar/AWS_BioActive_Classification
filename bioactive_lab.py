import os, sys
dirpath = os.getcwd()
sys.path.insert(0, dirpath + '/goal_tether_functions')
sys.path.insert(0, dirpath + '/predictive_modelers')
sys.path.insert(0, dirpath + '/predictive_modelers/assessment_resources')
sys.path.insert(0, dirpath + '/active_learners')
sys.path.insert(0, dirpath + '/data_acquisition')
sys.path.insert(0, dirpath + '/diagnostics')
from createCampaign_battleship import main as createCampaign
# from createImageCampaign_Bria import main as createCampaign
from runCampaign2 import main as runCampaign
from database import *
import outputManager
import time
import boto3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# Part 1 Plotting Function
def plot_simulation_accuracy(acc, title, mul_accuracy=False):
    fig, ax = plt.subplots()
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Iterations")
    ax.set_title(title)
    if mul_accuracy:
        ax.plot(np.arange(len(acc[0])), acc[0], label="Full Space")
        ax.plot(np.arange(len(acc[1])), acc[1], label="Forward Modeling")
        ax.plot(np.arange(len(acc[2])), acc[2], label="Prediction Only")
    else:
        ax.plot(np.arange(len(acc)), acc)
    ax.legend()
    plt.show()


def average_arrays(mat):
    array = []
    for i in range(25):
        avg = 0
        for m in range(len(mat)):
            if len(mat[m]) < i:
                continue
            avg += mat[m][i]
        avg = avg/len(mat)
        array.append(avg)
    return array

wd =os.getcwd()
print("Current Working Directory: ", wd)
print()

if path.exists("data/data.csv") is False:
    print("Retrieving Data from S3")

# read data from S3
s3 = boto3.resource('s3')
s3.Bucket('whatyouknowaboutmybucket').download_file('data.csv', wd + '/data/data.csv')

if path.exists("data/data.csv") is False:
    print("Retrieving Data from S3")
    time.sleep(5)

data = pd.read_csv("data/data.csv").dropna().to_numpy()
features = data[:, 4:]
labels = data[:, 2]

l = LabelEncoder()
labels = l.fit_transform(labels)
print(l.classes_)

s = KMeans(n_clusters=5)
# s.decision_function(features[:1000])
s.fit_transform(features[:1500])
print(s.score(features[1500:]))

d = np.zeros((20,20))

# create groundTruth
for i in range(len(data)):
    if data[i][0] - 1 >= len(d) or data[i][1] >= len(d[0]):
        continue
    d[data[i][0]-1][data[i][1]-1] = s.predict(features[i].reshape(1,-1))

print(d)


np.savetxt('data_acquisition/project.txt', d)


print(labels)






# exit()
'''
campaign = createCampaign()
runCampaign(campaign)
acc = [np.array(campaign.accuracy_full), np.array(campaign.accuracy_forwardModeling),
       np.array(campaign.accuracy_onlyPredictions)]

plot_simulation_accuracy(acc, "Model Accuracies for a Single Simulation", mul_accuracy=True)
'''

# Part 2 of Assignment - 2 independent variables (0-20) and 1 dependent variable (0-10) for 20 simulations

acc = []
for i in range(1):
    campaign = createCampaign()
    campaign.randoseed = 2
    # campaign.ESS.iVars = [('int', 0, 9), ('int', 0, 9)]
    # campaign.ESS.dVars = [('int', 0, 2)]
    campaign.groundtruthData = 'data_acquisition/project.txt'
    campaign.simsFlag = True
    runCampaign(campaign)
    acc = [campaign.accuracy_full, campaign.accuracy_forwardModeling, campaign.accuracy_onlyPredictions]
# acc = average_arrays(acc)
plot_simulation_accuracy(acc, "Three Accuracies for the Experimental Space", mul_accuracy=True)


# Part 3 of Assignment -
# acc1, acc2, acc3, acc4 = [], [], [], []
# for i in range(5):
#     campaign = createCampaign()
#     campaign.ESS.high_homogeneity = True
#     campaign.ESS.h_num = 2
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 2)]
#     campaign.ESS.dimarr = [20,20]
#     runCampaign(campaign)
#     acc = campaign.accuracy_onlyPredictions
#     acc1.append(acc)
#
# for i in range(5):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.h_num = 2
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 2)]
#     runCampaign(campaign)
#     acc = campaign.accuracy_onlyPredictions
#     acc2.append(acc)
#
# for i in range(5):
#     campaign = createCampaign()
#     campaign.ESS.high_homogeneity = True
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20,20]
#     runCampaign(campaign)
#     acc = campaign.accuracy_onlyPredictions
#     acc3.append(acc)
#
# for i in range(5):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20,20]
#     runCampaign(campaign)
#     acc = campaign.accuracy_onlyPredictions
#     acc4.append(acc)
#
# acc1, acc2, acc3, acc4 = average_arrays(acc1), average_arrays(acc2), average_arrays(acc3), average_arrays(acc4)
#
# plt.plot([i+1 for i in range(len(acc1))], acc1, label="H-2", color="blue")
# plt.plot([i+1 for i in range(len(acc2))], acc2, label="L-2", color="green")
# plt.plot([i+1 for i in range(len(acc3))], acc3, label="H-10", color="red")
# plt.plot([i+1 for i in range(len(acc4))], acc4, label="L-10", color="black")
# plt.ylabel("Accuracy (%)")
# plt.xlabel("Iterations")
# plt.title("Different Homogeneity within Experimental Spaces")
# plt.legend()
# plt.show()


# Part 4 of Assignment -

# acc1, acc2, acc3, acc4 = [], [], [], []
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0
#     campaign.randoseed= 45
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc = campaign.accuracy_onlyPredictions
#     acc1.append(acc)
#
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.randoseed = 1
#     campaign.ESS.error = 0.1
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc = campaign.accuracy_onlyPredictions
#     acc2.append(acc)
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0.5
#     campaign.randoseed = 2
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc = campaign.accuracy_onlyPredictions
#     acc3.append(acc)
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 1.0
#     campaign.randoseed=3
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc = campaign.accuracy_onlyPredictions
#     acc4.append(acc)
#
# acc1, acc2, acc3, acc4 = average_arrays(acc1), average_arrays(acc2), average_arrays(acc3), average_arrays(acc4)
#
# plt.plot([i+1 for i in range(len(acc1))], acc1, label="0.0", color="blue")
# plt.plot([i+1 for i in range(len(acc2))], acc2, label="0.1", color="green")
# plt.plot([i+1 for i in range(len(acc3))], acc3, label="0.5", color="red")
# plt.plot([i+1 for i in range(len(acc4))], acc4, label="1.0", color="black")
# plt.ylabel("Accuracy (%)")
# plt.xlabel("Iterations")
# plt.title("Different Error Rates within Experimental Spaces")
# plt.legend()
# plt.show()


# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0
#     campaign.randoseed = 53
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc1 = campaign.accuracy_onlyPredictions
#
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0
#     campaign.randoseed = 39
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc2 = campaign.accuracy_onlyPredictions
#
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0.1
#     campaign.randoseed = 32
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc3 = campaign.accuracy_onlyPredictions
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0.1
#     campaign.randoseed = 17
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc4 = campaign.accuracy_onlyPredictions
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0.5
#     campaign.randoseed = 3
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc5 = campaign.accuracy_onlyPredictions
#
# for i in range(1):
#     campaign = createCampaign()
#     campaign.ESS.low_homogeneity = True
#     campaign.ESS.error = True
#     campaign.ESS.error = 0.5
#     campaign.randoseed = 15
#     campaign.ESS.h_num = 10
#     campaign.ESS.iVars = [('int', 0, 19), ('int', 0, 19)]
#     campaign.ESS.dVars = [('int', 0, 9)]
#     campaign.ESS.dimarr = [20, 20]
#     runCampaign(campaign)
#     print(campaign.groundTruth)
#     acc6 = campaign.accuracy_onlyPredictions
#
#
# plt.plot([i+1 for i in range(len(acc1))], acc1, label="0.0 - B", color="blue")
# plt.plot([i+1 for i in range(len(acc2))], acc2, label="0.0 - N", color="green")
# plt.plot([i+1 for i in range(len(acc3))], acc3, label="0.1 - B", color="red")
# plt.plot([i+1 for i in range(len(acc4))], acc4, label="0.1 - N", color="black")
# plt.plot([i+1 for i in range(len(acc5))], acc5, label="0.5 - B", color="yellow")
# plt.plot([i+1 for i in range(len(acc6))], acc6, label="0.5 - N", color="cyan")
# plt.ylabel("Accuracy (%)")
# plt.xlabel("Iterations")
# plt.title("Different Categorical Models within Experimental Spaces")
# plt.legend()
# plt.show()
