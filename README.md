# AnomalyDetectionTS
This page provides datasets, codes and detailed experimental settings in paper "A New Distributional Treatment for Time Series and An Anomaly Detection Investigation" by Kai Ming Ting, Zongyou Liu, Hang Zhang, and Ye Zhu

In document "parameter.csv", each time series starts from period 0. 

Note that all time series are periodic and the period length of each can be easily known from the prior knowledge about data, e.g., the frequency of sampling in data collection. 
The period of some datasets vary slightly but it has no effect on the result of algorithms. Our algorithm works well when choosing subsequence length to be roughly the length of the period.

The parameter of some datasets should be specially explained.

dutch\_pwrdemand: This time series has power consumption for a Dutch research facility for the year 1997 (one power measurement every 15 minutes for 365 days). It shows a characteristic weekly pattern that consists of 5 power usage peaks corresponding to the 5 weekdays followed by 2 days of low power usage on the weekends. Anomalous weeks occur when one or more of the normal usage peaks during a week do not occur due to holidays. We can easily figure out that there are 672 points in each week(672=7*24*60/15) so the period length is 672.  It is Wednesday on January 1st so we let each week period start from Wednesday. There are total of 6 anomalous weeks. Some papers use this dataset with less anomalies because they treat continous anomalous weeks as 1 anomaly.  
J. J. Van Wijk and E. R. Van Selow, "Cluster and calendar based visualization of time series data," Proceedings 1999 IEEE Symposium on Information Visualization (InfoVis'99), 1999, pp. 4-9, doi: 10.1109/INFVIS.1999.801851.


