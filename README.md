# AnomalyDetectionTS
This page provides datasets, codes and detailed experimental settings in paper "A New Distributional Treatment for Time Series and An Anomaly Detection Investigation" by Kai Ming Ting, Zongyou Liu, Hang Zhang, and Ye Zhu

In document "parameter.csv", each time series starts from period 0. 

Note that all time series are periodic and the period length of each can be easily known from the prior knowledge about data, e.g., the frequency of sampling in data collection. 
The period of some datasets vary slightly but it has no effect on the result of algorithms. Our algorithm works well when choosing subsequence length to be roughly the length of the period.

The parameters of some datasets should be specially explained.

### dutch\_pwrdemand
This time series has power consumption for a Dutch research facility for the year 1997 (one power measurement every 15 minutes for 365 days). It shows a characteristic weekly pattern that consists of 5 power usage peaks corresponding to the 5 weekdays followed by 2 days of low power usage on the weekends. Anomalous weeks occur when one or more of the normal usage peaks during a week do not occur due to holidays. We can easily figure out that there are 672 points in each week(672=7x24x60/15) so the period length is 672.  It is Wednesday on January 1st so we let each week period start from Wednesday. There are a total of 6 anomalous weeks. Some papers use this dataset with less anomalies because they treat continous anomalous weeks as 1 anomaly.  The detailed information about this dataset can be seen in the paper below.  
J. J. Van Wijk and E. R. Van Selow, "Cluster and calendar based visualization of time series data," Proceedings 1999 IEEE Symposium on Information Visualization (InfoVis'99), 1999, pp. 4-9, doi: 10.1109/INFVIS.1999.801851.

### ann_gun
The anomalous period 2 is annotated by Keogh's work [https://www.cs.ucr.edu/~eamonn/discords/ICDM05_discords.pdf] as the first picture below shows. Other anomalous period subsequences which are proposed in Boniol's [Boniol Paul, et al. "Unsupervised and scalable subsequence anomaly detection in large data series." The VLDB Journal (2021): 1-23.] can be seen in the second picture below.

![image](https://user-images.githubusercontent.com/90513919/132955327-475b68cb-be6f-4400-bc35-cb75d1be208c.png)&nbsp;&nbsp;
![image](https://user-images.githubusercontent.com/90513919/132955329-80274222-f27e-4dc9-be6f-34683031c44e.png)



### Patient_respiration
We use the subset of dataset nprs44, beginning at 15500 and ending at 22000. There are one apparent anomaly and one subtle anomaly in this dataset as shown in the picture(cited from https://www.cs.ucr.edu/~eamonn/discords/ICDM05_discords.pdf).

![image](https://user-images.githubusercontent.com/90513919/132955446-d1d83ae5-c456-4897-9b6e-c9cc3122382d.png)


### TEK
Inspired by Jones Michael's work [Jones Michael, et al. "Exemplar learning for extremely efficient anomaly detection in real-valued time series." Data mining and knowledge discovery 30.6 (2016): 1427-1454.], we also concatenate dataset TEK14,TEK16 and TEK17 as TEK of length 15000. In Keogh's work [https://www.cs.ucr.edu/~eamonn/discords/ICDM05_discords.pdf], a total of 4 anomalies are marked. But in TEK14, 2 anomalous snippets belongs to the same period as shown in the picture below(cited from https://www.cs.ucr.edu/~eamonn/discords/ICDM05_discords.pdf). Since we detect anomalous subsequences by the unit of period, we treat it as one anomalous period subsequence of length 1000. So there are 3 anomalies in our annotations of this dataset.

![image](https://user-images.githubusercontent.com/90513919/132956852-64b15a89-b553-44f6-bfd3-fadb8c243873.png)

### MBA803,MBA805,MBA806,MBA820,MBA14046
We use the subset of first 100K points from the full datasets of each like Boniol does in the paper [Boniol Paul, et al. "Unsupervised and scalable subsequence anomaly detection in large data series." The VLDB Journal (2021): 1-23.].

