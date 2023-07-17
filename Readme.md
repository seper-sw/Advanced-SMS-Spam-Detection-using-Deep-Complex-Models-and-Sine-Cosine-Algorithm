
# Spam SMS detection

In this project i used KNN & Random-forest & SVM & MLP algorithms to detect spam from not-spam SMS and complete code exist in :
complete.ipynb 

I used the feature extraction phase that this part utilizes, which incorporates the metaheuristic algorithm SCA. It has been implemented in sca.py






spam.csv is my dataset

function.py is source of ANN that used in SCA to calculate loss function 
and  The fowchart of the proposed method for spam detection is :

<div align="left">
<img width="750" alt="flow-chart" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/5e736daf-833e-43d3-9e6e-bc69f63fd78a">
</div>





## Requirement

* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Tensorflow


## Brief review of SCA-algorithm for feature extraction
SCA algorithm use this formula to update feature vector X:

<div align="left">
<img width="600" alt="sca" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/1a3fd0f3-c162-4e32-aa14-36899b7e1b06">
</div>

and the cost function is :



<div align="left">
<img width="274" alt="cost-func" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/44d81731-a115-4a08-8669-3f555950d813">
</div>

S.T:


<div align="left">
<img width="81" alt="ST" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/bd4816bd-a624-469c-ae9d-63259f51e09a">
</div>
That implemention of this is in the sca.py


## Result
1-SCA helped to reduce 4409 features in TF-IDF method & 4404 featuers in BOW method that in each of them with a good results

2-After implementing and executing the necessary steps, the following results were obtained :

<div align="left">
<img width="758" alt="result" src="https://github.com/seper-sw/Spam-sms-detection/assets/94066230/ae64d75f-238b-49ca-8430-b781b68389b2">
</div>



## Refrence
1-SMS Spam Using Machine-Learning Algorithms by Fatima Zohra El Hlouli, Jamal Riffi, Mohamed Adnane Mahraz,
Ali El Yahyaouy and Hamid Tairi  


 2-Spam detection through feature selection using artifcial neural 
network and sine–cosine algorithm by Rozita Talaei Pashiri1
 · Yaser Rostami1  · Mohsen Mahrami1

 3-spam dataset from :
 kaggle datasets download -d uciml/sms-spam-collection-dataset
