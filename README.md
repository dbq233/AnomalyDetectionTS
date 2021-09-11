# AnomalyDetectionTS
This page provides datasets, codes and detailed experimental settings in paper "A New Distributional Treatment for Time Series and An Anomaly Detection Investigation" by Kai Ming Ting, Zongyou Liu, Hang Zhang, and Ye Zhu

In document "parameter.csv", each time series starts from period 0. 

Note that all time series are periodic and the period length of each can be easily known from the prior knowledge about data, e.g., the frequency of sampling in data collection. 
The period of some datasets vary slightly but it has no effect on the result of algorithms. Our algorithm works well when choosing subsequence length to be roughly the length of the period.

The parameter of some datasets should be specially explained.

dutch\_pwrdemand: 



