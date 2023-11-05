
#  SMS Spam Detection

In this project MLP & Complex MLP & KNN & Random-Forest & SVM Algorithms were used to detect spam from not-spam SMS And the Sine-Cosine Algorithm (SCA) was used for dimensionality reduction.
Complete code implementation already exists  in complete.py .

### complete.py includes :
1. Data Pre-processing :
   - Change each word to lower case
   - Removing stop words
   - Stemming
1. Tokenization with two methods:
   - TF-IDF
   -  BOW
1. Splitting Data set to Train and Test Set : 
   - 70% for Training.
   - 30% for Test.
1. Dimension Reduction Phase that used the SCA Algorithm.
1. Implementation of algorithms.  
 
### In the Dimension Reduction phase, we used the metaheuristic algorithm SCA. It has been implemented in sca.py
function.py is source of ANN that used in SCA to calculate loss function 


### spam.csv is our dataset








## Requirement

* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Tensorflow


## Brief review of SCA-algorithm 
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
<img width="495" alt="Screenshot 2023-11-06 024114" src="https://github.com/seper-sw/SMS-Spam-Detection/assets/94066230/8cda96f9-40c7-416a-b385-e50df6d8b27c">
</div>




## Refrence
1-SMS Spam Using Machine-Learning Algorithms by Fatima Zohra El Hlouli, Jamal Riffi, Mohamed Adnane Mahraz,
Ali El Yahyaouy and Hamid Tairi  


 2-Spam detection through feature selection using artifcial neural 
network and sine–cosine algorithm by Rozita Talaei Pashiri1
 · Yaser Rostami1  · Mohsen Mahrami1

 3-spam dataset from :
 kaggle datasets download -d uciml/sms-spam-collection-dataset
