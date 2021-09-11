import numpy as np
import random

from IDK import IDK


def IDK_T(X, t, psi,width,psi2):
    # # distance between every two points
    # distance_matrix = np.zeros((X.shape[0], X.shape[0]))
    # feature map of D/width


    window_num = (int)(np.ceil(X.shape[0]/width))

    featuremap_count = np.zeros((window_num,t * psi))
    all_count=np.zeros(t*psi)
    # onepoint_matrix[i]记录第i个点map到的t*psi维向量里哪些位置为1，onepoint_matrix[i][j]: area number of the i-th point in the j-th partition
    # 如果为初始值-1则表示该点在第time次映射到全0向量
    onepoint_matrix = np.full((X.shape[0], t), -1)
    pre_scores = np.zeros(X.shape[0])
    pre_scores_cmp=np.zeros(X.shape[0])
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[0]):
    #         if i < j:
    #             distance_matrix[i][j] = np.linalg.norm(X[i] - X[j])
    #         else:
    #             distance_matrix[i][j] = distance_matrix[j][i]
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(X.shape[0])]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        # distance between sample
        tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)

        # for i in range(len(sample_list)):
        #     for j in range(len(sample_list)):
        #         if i != j:
        #             if radius_list[i] == 0:
        #                 radius_list[i] = distance_matrix[sample_list[i]][sample_list[j]]
        #
        #             elif radius_list[i] > distance_matrix[sample_list[i]][sample_list[j]]:
        #                 radius_list[i] = distance_matrix[sample_list[i]][sample_list[j]]
        sample2sample[sample2sample < 1e-9] = 99999999;
        radius_list=np.min(sample2sample,axis=1)#每行的最小值形成一个行向量

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape)) #n*psi
        tem2 =np.dot(np.ones(X.shape),np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T) #n*psi
        min_dist_point2sample=np.argmin(point2sample,axis=1)#index
        #min_dist_point2sample_val = np.argmin(point2sample, axis=1)


        # map all points
        # for i in range(X.shape[0]):
        #     for j in range(len(sample_list)):
        #         if distance_matrix[i][sample_list[j]] < radius_list[j]:
        #             if onepoint_matrix[i][time] == -1:
        #                 onepoint_matrix[i][time] = j + time * psi
        #             elif distance_matrix[i][sample_list[j]] < distance_matrix[i][
        #                 sample_list[onepoint_matrix[i][time] - time * psi]]:
        #                 onepoint_matrix[i][time] = j + time * psi
        #     if onepoint_matrix[i][time] != -1:
        #         featuremap_count[onepoint_matrix[i][time]] += 1
        for i in range(X.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time]=min_dist_point2sample[i]+time*psi
                featuremap_count[(int)(i/width)][onepoint_matrix[i][time]] += 1
                all_count[onepoint_matrix[i][time]]+=1



    # feature map of D/width
    for i in range((int)(X.shape[0]/width)):
        featuremap_count[i] /= width
    isextra=X.shape[0] -(int)(X.shape[0] / width) * width
    if isextra>0:
        featuremap_count[-1] /= isextra

    all_count/=X.shape[0]
    score_of_windows = []

    #最后一个不完整的窗口去掉
    if isextra>0:
        featuremap_count=np.delete(featuremap_count,[featuremap_count.shape[0]-1],axis=0)

    #bitrepresentation


    # # cal feature map of every point
    # for i in range(onepoint_matrix.shape[0]):
    #     for ele in onepoint_matrix[i]:
    #         if ele != -1:
    #             pre_scores[i] += featuremap_count[(int)(i/width)][ele]
    #             pre_scores_cmp[i] += all_count[ele]
   
    return IDK(featuremap_count, t, psi2)

