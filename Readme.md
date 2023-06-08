
# Implementing the combination of two Machine Learning papers

In this project i used KNN & Random-forest & SVM & MLP algorithms to detect spam from not-spam SMS and complete code exist in :
complete.ipynb 


spam.csv is my dataset

function.py is source of ANN that used in SCA for selecting features 
and  The fowchart of the proposed method for spam detection is :

<div align="left">
<img width="755" alt="flow" src="https://github.com/seper-sw/sms/assets/94066230/d91d8369-ce70-491a-a5aa-9f7bbd811100">
</div>




## Required libraries
```bash
 pip install pandas
 pip install sklearn
 pip install nltk
 pip install numpy
```


## Brief review of first paper

In this article, KNN, Random-forest & SVM and MLP algorithms are used to identify spam SMS and TF-IDF and BOW methods are used on each algorithm.


## Brief review of second paper
This paper used SCA algorithm for feature extraction which this algorithm use this formula:

<div align="left">
<img width="755" alt="formol" src="https://github.com/seper-sw/sms-spam-detection/assets/94066230/12b0c69e-8fd4-46c1-99d5-1fb9ea3c38d4">
</div>

That implemention of this is in the sca.py
## Result
After implementing and executing the necessary steps, the following results were obtained :

<div align="left">
<img width="758" alt="ml" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/9adf5b33-fac3-41ac-88fb-e5396928ec69">
</div>
## Refrence
1-SMS Spam Using Machine-Learning Algorithms by Fatima Zohra El Hlouli, Jamal Riffi, Mohamed Adnane Mahraz,
Ali El Yahyaouy and Hamid Tairi  

 then i used SCA(Sine-Cosine-algorithm) to extract most important features that it's actually implemention of this paper:

 2-Spam detection through feature selection using artifcial neural 
network and sine–cosine algorithm by Rozita Talaei Pashiri1
 · Yaser Rostami1  · Mohsen Mahrami1

 3-spam dataset from :
 kaggle datasets download -d uciml/sms-spam-collection-dataset
