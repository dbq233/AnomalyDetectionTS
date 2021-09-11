import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.svm import OneClassSVM

from IDK_T import IDK_T
from sklearn.preprocessing import StandardScaler

from readNormaTXT import get_IDKpAtk, get_scoreCycleList_from_sliding, get_slidingpAtk, get_ACC
import ot

def getDiscordList_ifo(df, cycle, bin=10):
    #    df = pd.read_csv(filename, header=None)
    #    n = len(df)
    #    dff = np.array(df).astype(int).reshape(n).tolist()
    #    df = []
    #    #n = min(n, 100000)
    #    for i in range(n):
    #        df.append([dff[i]])

    data = []
    n = len(df)
    # df=df.reshape(n)
    for i in range(n - cycle + 1):
        subsequence = df[i:i + cycle]

        # subsequence = preprocessing.scale(subsequence)
        data.append(subsequence)

    return np.array(data)


def getDiscordList_ocsvm(df, cycle, bin=10):
    #    df = pd.read_csv(filename, header=None)
    #    n = len(df)
    #    dff = np.array(df).astype(int).reshape(n).tolist()
    #    df = []
    #    #n = min(n, 100000)
    #    for i in range(n):
    #        df.append([dff[i]])

    data = []
    n = len(df)
    for i in range(int(n / cycle)):
        subsequence = df[i * cycle:(i + 1) * cycle]

        # subsequence = preprocessing.scale(subsequence)
        # data.append(subsequence)

        max_abs_scaler = preprocessing.MaxAbsScaler()
        subsequence = max_abs_scaler.fit_transform(subsequence)
        hist, bin_edges = np.histogram(subsequence, bins=bin)
        data.append(hist / np.sum(hist))

    return np.array(data)


def Wasserstein_distance(label_sequences, sinkhorn=False, sinkhorn_lambda=1e-2, bin=10):
    """
    Generate the Wasserstein distance matrix for the subsequences
    """
    n = len(label_sequences)
    M = np.zeros((n, n))
    nbins = bin
    xxx = np.arange(nbins, dtype=np.float64)
    costs = ot.dist(xxx.reshape((nbins, 1)), xxx.reshape((nbins, 1)))
    costs /= costs.max()
    for subseque_index_1, subseque_1 in enumerate(label_sequences):
        for subseque_index_2, subseque_2 in enumerate(label_sequences[subseque_index_1:]):
            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(subseque_1)) / len(subseque_1),
                                  np.ones(len(subseque_2)) / len(subseque_2), costs, sinkhorn_lambda,
                                  numItermax=50)
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = \
                    ot.emd2(subseque_1, subseque_2, costs)
    M = (M + M.T)
    return M


def wwl(list_of_distributions, sinkhorn=False, sinkhorn_lambda=1e-2, gamma=None, bin=10):
    """
    using laplacian_kernel ,,, cost matrix
    return kernel matrix of shape (n_distributions, n_distributions)
    """
    D_W = Wasserstein_distance(list_of_distributions, sinkhorn, sinkhorn_lambda, bin=bin)
    # wwl = laplacian_kernel(D_W, gamma=gamma)

    wwl = np.exp(-D_W / gamma)
    return wwl


def ocsvm_WL(gamma, sinkhorn_lambda, df, cycle, bin=10):
    allTdata = getDiscordList_ocsvm(df, cycle, bin=bin)
    X_WL = wwl(allTdata, sinkhorn=False, sinkhorn_lambda=sinkhorn_lambda, gamma=gamma, bin=bin)
    ocsvm = OneClassSVM(gamma='auto', kernel='precomputed')
    ocsvm.fit(X_WL)

    test_scores = ocsvm.score_samples(X_WL)

    return test_scores


def ifor(psi, X, cycle):

    allTdata = getDiscordList_ifo(X, cycle)
    n = len(allTdata)
    psi = min(psi, n)
    clf = IsolationForest(n_estimators=100, max_samples=psi)

    clf.fit(allTdata)


    test_scores = -clf.score_samples(allTdata)  # yue xiao yue yi chang

    return test_scores
    # test_scores = get_scoreCycleList_from_sliding(cycle, cycle, n+cycle-1, test_scores, int(cycle/2))


#k is anomaly_cycles length
def get_acc(k, value_list, anomaly_cycles):
    true_pos = 0
    sorted_value_list = np.argsort(value_list)
    for it in sorted_value_list[0:k]:
        if it in anomaly_cycles:
            true_pos += 1
    count = k;
    for index in range(k, len(sorted_value_list)):
        if value_list[sorted_value_list[index]] != value_list[sorted_value_list[k-1]]:
            break
        count+=1
        if sorted_value_list[index] in anomaly_cycles:
            true_pos+=1
    return true_pos / count





def get_label(X, cycle, anomaly_list):
    window_num = (int)(X.shape[0] / cycle)
    label_list = np.zeros(window_num)
    label_list[anomaly_list] = 1
    return label_list


def reshapeSubsequence(X, width):
    if X.shape[0] % width != 0: print("不能整除划分")

    window_num = (int)(X.shape[0] / width)
    X = X[0:window_num * width]
    return X.reshape((window_num, width))


def subsequenceFluc(X, width):
    if X.shape[0] % width != 0: print("不能整除划分")

    window_num = (int)(X.shape[0] / width)

    fluc_list = np.zeros((window_num, width - 1))
    for i in range(0, window_num * width):
        if i % width != 0:
            fluc_list[(int)(i / width)][(i % width) - 1] = X[i] - X[i - 1]
    return fluc_list


def subsequenceMinMaxScale(X, cycle):
    lo = 0
    hi = cycle
    scaler = preprocessing.MinMaxScaler()
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        X[lo:hi] = scaler.fit_transform(X[lo:hi])
        lo = hi
        hi += cycle


def prepocessSubsequence(X, cycle):
    lo = 0
    hi = cycle
    scaler = StandardScaler()
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        scaler = scaler.fit(X[lo:hi])
        X[lo:hi] = scaler.transform(X[lo:hi])
        lo = hi
        hi += cycle


def findthredsomeInSlidingWindowIDK(scorelist, redlist, width):
    mini = 1;
    for i in range(1, len(redlist)):
        if redlist[i][0] - redlist[i - 1][1] >= width:
            tem = scorelist[range(redlist[i - 1][1], redlist[i][0] - width + 1)]
            mini = min(mini, min(tem))
    return mini


def findthredsomeInCycleIDK(scorelist, anomaly_cycle):
    tem = np.delete(scorelist, anomaly_cycle, axis=0)
    return min(tem)

def getDiscordList(filename):
    df = pd.read_csv(filename, header=None)
    df = np.array(df).astype(int).reshape(len(df))
    return df

def drawIDK_T_discords(TS, idk_scores, number=3):
    sorted_index = np.argsort(idk_scores)
    color_list = ['r', 'y', 'b', 'g', 'c', 'm', 'k']  # c天蓝,m紫色,k黑色
    cur = 0
    plt.plot(TS)
    for index in sorted_index:
        if number <= 0:
            break;
        number -= 1
        ls = range(index * cycle, (int)(index * cycle + cycle))
        ly = TS[ls]
        plt.plot(ls, ly, color=color_list[cur])
        cur = (cur + 1) % 7
    plt.show()


cycle=300
anomaly_cycles = [5,10,20,30]


df = pd.read_csv("Discords_Data/noisysine_small_test.txt", header=None)

df = np.array(df).reshape(df.shape[0])

redlist = []
redlist.append((0, 0))
for it in anomaly_cycles:
    redlist.append((cycle * it, cycle * it + cycle))

redlist_index = []

mode = 0
start = -1
end = -1


redlist.append((len(df), len(df)))
#df = val1
# plt.plot(df, color='b')

# #增加数据
# df=df[0:3740]
# for i in range(10):
#     df=np.vstack((df,df[10:330]))


X = df.reshape((len(df), 1))
#X=X[0:100000]
#X=subsequenceFluc(X.reshape(X.shape[0]),cycle).reshape((-1,1))
print(X.shape)

##多维数据
# X= np.hstack((X,val2.reshape((len(val2),1))))
# print(X.shape)

# prepocessing

# min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.fit_transform(X)


# subsequence normalization
plt.plot(X, color='b')
plt.show()
TS = X.copy()
TS = TS.reshape(TS.shape[0])
#prepocessSubsequence(X,cycle)
#X=subsequenceFluc(X,X.shape[0]).reshape(-1,1)

# subsequenceMinMaxScale(X,cycle)


plt.plot(X, color='b')
for it in redlist:
    ls = range(it[0], it[1])
    ly = X[ls]
    plt.plot(ls, ly, color='r')

plt.show()


# for cycle in range(10):
#     psi = 1
#     for i in range(12):
#         psi *= 2
#
#         if psi > X.shape[0]:
#             break
#
#         pre_scores_list = IDK_twice(X, t, psi,400)
#         pd=1
#         for p in pre_scores_list:
#             pd*=2
#             idk_index = np.argsort(p)
#             plt.title("cycle=%d,psi1=%d,psi2=%d" %(cycle,psi,pd))
#             thredsome=findthredsomeInSlidingWindowIDK(p,redlist,400)
#             line = np.full(len(p), thredsome)
#
#             plt.plot(p)
#             plt.plot(line, color='g')
#             plt.show()
t = 100
psi1 =2
psi2 =2
sum=0
acc=0

ocsvm_Wl_socres=ocsvm_WL(0.01,1e-2,X,cycle,bin=10)
print(roc_auc_score(get_label(X, cycle, anomaly_cycles), -ocsvm_Wl_socres))
plt.plot(ocsvm_Wl_socres)
thredsome = findthredsomeInCycleIDK(ocsvm_Wl_socres, anomaly_cycles);
line = np.full(len(ocsvm_Wl_socres), thredsome)
plt.plot(line,color='g')
plt.show()
for i in range(10):
    p = IDK_T(X, t=100, psi=8, width=cycle,psi2=4)


    # bfed_list=DTWDiscordsDiscovery(X,256,10)
    # print(np.argmax(bfed_list))
    # pre_scores_list=np.array(pre_scores_list)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # pre_scores_list = min_max_scaler.fit_transform(pre_scores_list.T).T

    # trend_score = calTrend(X, cycle)
    # trend_score_list = 5 * calTrendIDK(X, cycle)
    # fluctuate_score=calFluctuateOfPointFeatureSpace(idk_t[1],cycle)



    idk_index = np.argsort(p)
    plt.title("psi1=%d,psi2=%d" % (psi1, psi2))
    plt.plot(p, color='b')
    thredsome = findthredsomeInCycleIDK(p, anomaly_cycles);
    line = np.full(len(p), thredsome)
    plt.plot(line, color='g')
    auc=roc_auc_score(get_label(X, cycle, anomaly_cycles), -p)
    sum+=auc
    #acc+=get_ACC(-p,anomaly_cycles,len(anomaly_cycles))

    print(auc)
    plt.show()
        #drawIDK_T_discords(TS, p, len(anomaly_cycles))

        # plt.plot(p-trend_score, color='r')
        # thredsome = findthredsomeInCycleIDK(p-trend_score, anomaly_cycles);
        # line = np.full(len(p), thredsome)
        # plt.plot(line, color='y')

print("AUC",sum/10)

# for p in trend_score_list:
#     pd*=2
#     plt.title("psi= %d" % (pd))
#     plt.plot(p, color='b')
#     thredsome = findthredsomeInCycleIDK(p, anomaly_cycles);
#     line = np.full(len(p), thredsome)
#     plt.plot(line, color='g')
#     plt.plot(-trend_score, color='r')
#     thredsome = findthredsomeInCycleIDK(-trend_score, anomaly_cycles);
#     line=np.full(len(p),thredsome)
#     plt.plot(line, color='y')
#     plt.show()


# for p in range(5):
#     pd *= 2
#     plt.title("psi1=%d,k=%d" % (psi, pd))
#     for time in range(5):
#         tem=create_cluster(pre_scores_list, pd)
#         print(tem[1])
#         plt.plot(tem[0])
#         plt.show()
