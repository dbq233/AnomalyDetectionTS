import scipy.io as scio
from sklearn.preprocessing import StandardScaler

from IDK import *
from sklearn import *
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from IDK_T import IDK_T
import copy


# zone是允许sliding超出的点数，最小是0，即只取在cycle范围内的subsequence，不允许超；最大是win_size-1，即只有subsequence有一个点在cycle内都算是cycle内的
def get_scoreCycleList_from_sliding(cycle, win_size, labels, score_list, zone):
    zone = min(zone, win_size - 1)
    result = np.zeros(len(labels))
    for i in range(len(result)):
        lo = max(0, i * cycle - zone)
        if lo >= len(score_list):
            print("有结果为0的")
            break
        hi = min(len(score_list),
                 (i + 1) * cycle + zone - win_size + 1)  # score_list中的[lo,hi),即lo和hi索引的是subsequence起始位置
        assert lo < hi
        # if labels[i]==1:
        #     result[i]=np.max(score_list[lo:hi])
        # else:
        #     result[i]=np.min(score_list[lo:hi])
        result[i] = np.max(score_list[lo:hi])
    return result



#wrong
def get_acc_sliding(tok_k_idx, abnormal_index, total_length, win_size, zone):
    zone = min(zone, win_size - 1)
    point_label = np.zeros(total_length)
    for it in abnormal_index:
        lo = max(0, it - zone)
        hi = min(total_length, it + 1 + zone)
        point_label[lo:hi] = 1
    result = 0
    for it in tok_k_idx:
        if point_label[it] == 1:
            result += 1
    return result / len(tok_k_idx)


def get_ACC(anomaly_scorelist,anomaly_cycles,k):
    sorted_index=np.argsort(-anomaly_scorelist)
    i=k
    while i<len(anomaly_scorelist) and anomaly_scorelist[sorted_index[i]]==anomaly_scorelist[sorted_index[k-1]]:
        i+=1
    pos=0
    for index in sorted_index[0:i]:
        if index in anomaly_cycles:
            pos+=1
    return pos/i

def get_slidingpAtk(k, value_list, anomaly_area_list,win_size, zone=None):
    if zone==None:
        zone=(int)(win_size/2)
    zone = min(zone, win_size - 1)
    visited = np.zeros(len(anomaly_area_list))
    covered=np.zeros(len(value_list))
    valk = k
    true_pos = 0
    cur = 0
    sorted_value_list = np.argsort(-value_list)
    for cyc in sorted_value_list:
        if k == 0:
            break
        flag = False
        lo = cyc
        hi = cyc +win_size
        for i in range(lo, hi):
            for area_index in range(len(anomaly_area_list)):
                if anomaly_area_list[area_index][0] <= i < anomaly_area_list[area_index][1]:
                    flag = True
                    if visited[area_index] == 0:
                        true_pos += 1
                        k -= 1
                        visited[area_index] = 1
        if flag == False and covered[cyc]==0:
            k -= 1
            ll=max(0, cyc - zone)
            hh = min(len(value_list),
                     cyc + zone + 1)  # score_list中的[lo,hi),即lo和hi索引的是subsequence起始位置
            covered[ll:hh]=1

        cur += 1
    # 若k很大使得没有break，则循环结束cur出来时已经是len(value_list)
    rest_list = []
    # 若第k个有后续score相等的情况
    for i in range(cur, len(value_list)):
        if value_list[sorted_value_list[i]] == value_list[sorted_value_list[cur - 1]]:
            rest_list.append(sorted_value_list[i])
            valk += 1
        else:
            break
    #flagD用来标记是不是前面检测到的异常区域，flag用来标记是不是异常
    for cyc in rest_list:
        flagD = False
        flag = False
        lo = cyc
        hi = cyc + win_size
        for i in range(lo, hi):
            for area_index in range(len(anomaly_area_list)):
                if anomaly_area_list[area_index][0] <= i < anomaly_area_list[area_index][1]:
                    flag = True
                    if visited[area_index] == 0:
                        true_pos += 1
                        visited[area_index] = 1
                    else:
                        flagD=True
        #是之前检测到的异常或者是之前检测到的非异常隔离区内
        if flagD == True or (flag == False and covered[cyc] == 1):
            valk-=1
        if flag == False and covered[cyc] == 0:
            ll = max(0, cyc - zone)
            hh = min(len(value_list),
                     cyc + zone + 1)  # score_list中的[lo,hi),即lo和hi索引的是subsequence起始位置
            covered[ll:hh] = 1

    return true_pos / valk

def get_IDKpAtk(k, value_list, anomaly_area_list,cycle):
    visited=np.zeros(len(anomaly_area_list))
    valk=k
    true_pos = 0
    cur=0
    sorted_value_list = np.argsort(value_list)
    for cyc in sorted_value_list:
        if k==0:
            break
        flag = False
        lo=cyc*cycle
        hi=cyc*cycle+cycle
        for i in range(lo,hi):
            for area_index in range(len(anomaly_area_list)):
                if anomaly_area_list[area_index][0]<=i<anomaly_area_list[area_index][1]:
                    flag=True
                    if visited[area_index]==0:
                        true_pos+=1
                        k-=1
                        visited[area_index]=1
        if flag==False:
            k -= 1
        cur+=1
    #若k很大使得没有break，则循环结束cur出来时已经是len(value_list)
    rest_list=[]
    #若第k个有后续score相等的情况
    flag = False
    for i in range(cur,len(value_list)):
        if value_list[sorted_value_list[i]]==value_list[sorted_value_list[cur-1]]:
            rest_list.append(sorted_value_list[i])
            valk+=1
        else:
            break
    #flag用来标记是前面已经检测出的异常，这种就不需要增加valk了
    for cyc in rest_list:
        lo=cyc*cycle
        hi=cyc*cycle+cycle
        flag=False
        for i in range(lo,hi):
            for area_index in range(len(anomaly_area_list)):
                if anomaly_area_list[area_index][0]<=i<anomaly_area_list[area_index][1]:
                    if visited[area_index]==0:
                        true_pos+=1
                        visited[area_index]=1
                    else:
                        flag=True
        if flag==True:
            valk-=1








    return true_pos/valk


def getDiscordList(filename):
    df = pd.read_csv(filename, header=None)
    df = np.array(df).astype(int).reshape(len(df))
    return df


def MP_auc(ts, MP):
    pass

def get_label(X, cycle, anomaly_list):
    window_num = (int)(X.shape[0] / cycle)
    label_list = np.zeros(window_num)
    label_list[anomaly_list] = 1
    return label_list

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


def read_mat(dataFile):
    data = scio.loadmat(dataFile)
    data = data['score_list']
    return data


def findthredsomeInSlidingWindowIDK(scorelist, redlist, width):
    maxi = -np.inf;
    index = -1
    for i in range(1, len(redlist)):
        if redlist[i][0] - redlist[i - 1][1] >= width:
            tem = scorelist[range(redlist[i - 1][1], min(redlist[i][0] - width + 1, len(scorelist)))]
            # if max(tem)>maxi:
            #     maxi=max(tem)
            #     index=i
            maxi = max(maxi, max(tem))
    return maxi


def drawDiscords(TS, score_list, width, number=3):
    color_list = ['r', 'y', 'b', 'g', 'c', 'm', 'k']  # c天蓝,m紫色,k黑色
    sorted_score_list = np.argsort(-score_list)
    cur = 0
    top_k_idx = []
    top_k_idx.append(sorted_score_list[0])
    for i in range(1, len(sorted_score_list)):
        if len(top_k_idx) >= number:
            break
        candidate = True
        for jt in top_k_idx:
            if (abs(jt - sorted_score_list[i]) < width):
                candidate = False
                break
        if candidate:
            top_k_idx.append(sorted_score_list[i])
    # print(score_list[top_k_idx])
    # print(top_k_idx)
    # print(score_list[1500])

    plt.plot(TS)
    for index in top_k_idx:
        ls = range(index, (int)(index + width))
        ly = TS[ls]
        plt.plot(ls, ly, color=color_list[cur])
        cur = (cur + 1) % 7
    plt.show()
    return top_k_idx


if __name__ == '__main__':
    cycle = 1000
    anomaly_cycles = [4,7,11]  # 异常序列所在周期

    df = pd.read_csv("TEK_ALL.txt", header=None)

    # df = pd.read_csv("TEK_ALL.txt", header=None)
    dg = pd.read_csv("noisysine_small_test_gt.txt", header=None)
    df = np.array(df).reshape(df.shape[0])
    # df2 = np.array(df2).reshape(df2.shape[0])
    # df= np.concatenate((df,df2))
    dg = np.array(dg).reshape(dg.shape[0])
    dl = dg[0].split()

    # df =  pd.read_csv("ECG_data/xmitdb_x108_0.txt",header=None)
    #
    # print(type(df))
    # df=np.array(df)
    # print(df.shape)
    # dl=[]
    # for it in df:
    #     dl.append(it[0].split())
    # for i in range(len(dl)):
    #     for j in range(3):
    #         dl[i][j]=float(dl[i][j])
    # data=pd.DataFrame(dl,columns=["time","val1","val2"])
    # time=np.array(data["time"])
    # val1=np.array(data["val1"]).reshape(data.shape[0],1)
    # val2=np.array(data["val2"]).reshape(data.shape[0],1)

    ##video
    # df = pd.read_csv("video.txt", header=None)
    #
    # print(type(df))
    # df = np.array(df)
    #
    # dl = []
    # for it in df:
    #     dl.append(it[0].split())
    # for i in range(len(dl)):
    #     for j in range(2):
    #         dl[i][j] = float(dl[i][j])
    # data = pd.DataFrame(dl, columns=["val1", "val2"])
    # val1 = np.array(data["val1"]).reshape(data.shape[0], 1)
    # val2 = np.array(data["val2"]).reshape(data.shape[0], 1)

    redlist = []
    redlist.append((0, 0))
    # for it in anomaly_cycles:
    #     redlist.append((cycle * it, cycle * it + cycle))

    redlist_index = []
    # for i in range(len(df)):
    #     df[i] = float(df[i])
    # for i in range(len(dl)):
    #     dl[i] = float(dl[i])
    # df=df[40000:85000]
    # dl=dl[40000:85000]
    mode = 0
    start = -1
    end = -1
    # 整理出每个异常区域的范围,[start,end)

    # df=df[450:]
    # for i in range(len(dl)):
    #     if dl[i] == 1:
    #         if mode == 0:
    #             start = i
    #             mode = 1
    #     else:
    #         if mode == 1:
    #             mode = 0
    #             end = i
    #             redlist.append((start, end))
    # if mode == 1:
    #     redlist.append((start, len(dl)))

    #    df=val1
    # plt.plot(df, color='b')

    # #增加数据
    # df=df[0:3740]
    # for i in range(10):
    #     df=np.vstack((df,df[10:330]))

    X = df.reshape((len(df), 1))
    #X=X[0:100000]
    pos_list = []
    #pos_list=[4250,7100,11100,11450]
    # pos_list = getDiscordList(
    #     "RealDatasets/ANNOTATIONS/MBA_820(1-100K).txt")
    anomaly_length =1000
    # anomaly_area_list=[]
    # for it in pos_list:
    #     anomaly_area_list.append([it,it+75])
    # for pos in pos_list:
    #     #redlist.append((pos, pos + anomaly_length))
    #     tem = (int)(pos / cycle)
    #     next = (int)((pos + 50 - 1) / cycle)
    #
    #     for cc in range(tem, next + 1):
    #         if cc not in anomaly_cycles:
    #             anomaly_cycles.append(cc)
    #np.savetxt("MBA_820(1-100K)_cycle.txt",anomaly_cycles,"%d")
    for it in anomaly_cycles:
        redlist.append((cycle * it, cycle * it + cycle))
    redlist.append((len(X), len(X)))
    # 多维数据
    # X= np.hstack((X,val2.reshape((len(val2),1))))
    print(X.shape)

    # prepocessing

    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)

    # subsequence normalization
    plt.plot(X, color='b')
    plt.show()
    # prepocessSubsequence(X,cycle)
    # subsequenceMinMaxScale(X,cycle)

    plt.plot(X, color='b')
    for it in redlist:
        ls = range(it[0], it[1])
        ly = X[ls]
        plt.plot(ls, ly, color='r')

    plt.show()
    plt.subplot()

    data = pd.read_csv("NORMA_result/Norm_TEK_1000_1.2_sample.txt")
    dtw_list = np.array(data).reshape(len(data))
    #dtw_list = np.array(data['MP'])

    # nn_index = np.array(data['NN_index'])
    # dtw_list=np.insert(dtw_list,0,np.zeros(cycle-1))
    plt.plot(dtw_list)
    #thredsome = findthredsomeInSlidingWindowIDK(dtw_list, redlist, width=cycle)
    #line = np.full(len(dtw_list), thredsome)
    #plt.plot(line)
    plt.show()
    print(len(pos_list))
    # print(
    #     get_acc_sliding(drawDiscords(X, score_list=dtw_list, width=anomaly_length, number=len(anomaly_cycles)), get_label(X,cycle,anomaly_cycles), len(X),
    #                     anomaly_length, (int)(anomaly_length / 2)))

    drawDiscords(X, score_list=dtw_list, width=anomaly_length, number=3)
    Norm_Ano_score_per_cycle=get_scoreCycleList_from_sliding(cycle,anomaly_length,get_label(X,cycle,anomaly_cycles),dtw_list,(int)(anomaly_length/2))
    print(roc_auc_score(get_label(X,cycle,anomaly_cycles),Norm_Ano_score_per_cycle))
    print(get_slidingpAtk(len(pos_list),dtw_list,anomaly_area_list,anomaly_length))
    print(get_ACC(Norm_Ano_score_per_cycle,anomaly_cycles,len(anomaly_cycles)))
    # get_scoreCycleList_from_sliding()
    # plt.subplot(2,1,1)
    # plt.ylim(-1.5, 1.5)
    # plt.plot(range(200,301),X[9457+100:9457+201])
    # plt.subplot(2, 1, 2)
    # plt.ylim(-1.5,1.5)
    # plt.plot(range(200,301),X[5691+100:5691+201])
    # plt.show()
