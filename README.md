# Card Transaction Fraud Detection Report

## 1. Executive Summary
Card transaction fraud has always been a crucial concern by financial institutions and card owners. According to Nielson’s report in 2018, payment card fraud losses reached $28.65 billion worldwide. With the significant changes in payment methods and increase in transaction data in recent decades, entities try to utilize more technology and data machine learning knowledge to optimize the detection process.

This report looks into the credit card transaction data and explores supervised models to detect potential fraud and thereby avoid potential fraud loss. The source data involves card transactions from a government entity in 2010, including a million property records with 10 fields, such as card number, merchant description, merchant characteristics, transaction amount, etc. As part of our analysis, we would:

(a) Evaluate the data quality and perform preliminary analysis;

(b) Do data cleaning and delete non-purchasing data or enormous amounts of data, then fill up empty fields with appropriate values;

(c) Create over 600 expert variables and Z-scale variables to give them a standardized measurement basis;

(d) Separate the data into modeling part and out-of-time (“OOT”) part, and use the modeling data to pick the most critical variables for modeling by averaging the ranking result of Kolmogorov-Smirnov (“KS”) and fraud detection rate (“FDR”) at 3%? rejected rate;

(e) Build supervised fraud prediction models with 7 different algorithms (i.e. logistic regression, neural network, decision tree, boosted trees, random forest, support vector machines (“SVM”)) and tune hyperparameters across 10 runs for each iterate, get the FDR for training, test and OOT data;

(f) Summarize the final result based on the final model and its adjusted parameters;

(g) Review our model and give suggestions for future updates.

To find the best model, the variables were tested in 7 different models. We finally chose 10 variables applying to Neural Network algorithms. The model is effective in identifying fraudulent records and the average FDR at 3% on the OOT pool is 54.5%. The model can be further adjusted according to different scenarios by creating more expert features and tuning hyperparameters.

## 2. Description of data

The “card transactions.csv” is a dataset that contains actual credit card purchases from a US government organization. Each entry of the data has a label indicating whether the transaction is a fraud. The purpose of this dataset is to detect potential credit card transaction fraud. For more information on the data, please see Appendix I for reference.

● The data provided is “Actual Credit Card Purchases” from the US government. It provides credit card transaction identifying information including data like card number, transaction date, the amount of transaction, whether the transaction is fraudulent, etc.

● Where it came from: This data is actual credit card purchases from a US government organization.

● Time period it covers: This is a 12-month data covering the period from 2006-01-1 to 2006-12-31. And the data is sorted chronologically by “Recnum”.

● Number of fields: There are 10 columns/fields. The “Fraud” field is a dependent variable with a label indicating fraud (0 represents “no fraud”, and 1 represents “fraud”).

● Number of records: There are 96,753 records. Only 1,059 of them are frauds records.

### 2.1 Field Summary Tables
Numeric Fields Summary Table

<img width="854" alt="Screenshot 2023-10-09 at 10 39 59 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b53dc5ac-4647-4224-9e01-ade5130adc61">

Categorical Fields Summary Table

<img width="624" alt="Screenshot 2023-10-09 at 10 46 41 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/46fabc33-e780-4237-bc4f-50003383fa71">

### 2.2 Distribution of Fields

1.  Recnum
   
● This field is a unique identifier for each record, essentially a serial number.

● All the records are sorted chronologically by “Recnum”, which can be regarded as time order.

● Each row here has a unique non-null value.

2. Cardnum
   
● This field is a categorical variable classifying the credit card number of each transaction.

● There are 1,645 card numbers having fraud records, and the card number 5142148452 was defrauded the most times.

<img width="848" alt="Screenshot 2023-10-09 at 10 48 16 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/968a85b4-233d-4a85-b398-9a9641918d78">

3. Date
● This field represents when the credit card transaction happened.

● It covers a period of 12 months from January 1st, 2006 to December 31st, 2006, which means all the transactions are happened and evaluated in 2006.

● And all the records are sorted chronologically according to the transaction date.

● February 28th, 2006 saw the highest number of credit card transactions.

● From the weekly and monthly transaction plots we can see that there is a sharp drop after September. It is because the U.S. government has a fiscal year that starts on October 1st. So the end of September is also the end of the fiscal year for the government. It's the fact that this is a government purchase card that makes it different from consumers’ purchases. The government gets its budget to reset on October 1st. So it might be conservative and spend little at the beginning of a new fiscal year.

<img width="837" alt="Screenshot 2023-10-09 at 11 18 53 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/7d99184d-7356-4852-b12f-cf1f382cdcf3">

<img width="810" alt="Screenshot 2023-10-09 at 11 19 22 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b9b0738e-5be5-41bc-bd66-52bc4a0e204c">

<img width="852" alt="Screenshot 2023-10-09 at 11 21 14 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/fa352841-cbbc-4578-b1e8-62abb6dc678d">

<img width="865" alt="Screenshot 2023-10-09 at 11 21 36 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/08d64ad9-d40b-4a8f-83da-12f6297dedc7">

4. Merchnum
   
● This is a categorical field containing the merchant number of each transaction.

<img width="826" alt="Screenshot 2023-10-09 at 11 22 15 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/cb773c4d-e0b6-4066-a707-ddbdef32990f">

5. Merch description
● This field contains the description of each merchant.

<img width="826" alt="Screenshot 2023-10-09 at 11 22 52 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/4f029141-641b-4779-b03e-802bc3748549">

6. Merch state

● Merch state is a categorical variable representing the state of the merchant.

● Most of the merchants of these transactions are in Tennessee.

● There are 227 unique values for this field. Besides the abbreviation for 50 states in the US, there are many three-digit values, I infer that it’s the three-digit zip code of specific regions. There also exist some areas outside the US, like BC, QC, ON, etc.

<img width="829" alt="Screenshot 2023-10-09 at 11 24 22 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/1b8ee8f5-04b6-4868-865a-85ed8fad1ea9">

7. Merch zip
   
● This field contains the zip code of each merchant.

<img width="765" alt="Screenshot 2023-10-09 at 11 24 47 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/baa56db2-ca3f-4f84-9f6a-1c46450e35c7">

8. Transtype
   
● This field is a categorical variable that indicates the transaction type.

<img width="780" alt="Screenshot 2023-10-09 at 11 26 08 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/deb1054b-6f2a-4aec-9da1-3d0258178e28">

9. Amount
    
● This field is a numeric variable that indicates the amount of each transaction.

● The distribution is right-skewed. Most of the transaction amounts are under $2500. There are fewer large-amount transactions than small-amount transactions.

● From the distribution of the amount of fraud and no fraud records, we can see that generally there are more fraudulent transactions than non-fraudulent transactions when the transaction amount ranges from $500 to $13000.

● In the range between 13000 and 23000, overall, there are more non-fraudulent transactions than fraudulent transactions.

● We can infer that fraud often happens on transactions with extremely small amounts and transactions with amounts between 13000 and 23000.

<img width="721" alt="Screenshot 2023-10-09 at 11 27 05 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/be6f91dc-454d-4c92-915f-0bf356ffc815">

<img width="667" alt="Screenshot 2023-10-09 at 11 27 24 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/2bf4e83f-573d-4109-bef6-6859bd803e14">

10. Fraud

● This field is a categorical variable describing whether the transaction had fraud or not, where 0 represents no fraud and 1 represents fraud.

● There are very few frauds in these transactions.

<img width="876" alt="Screenshot 2023-10-09 at 11 28 29 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/381026e7-74fb-4154-8ff0-1fa710416ce1">

## 3. Data Cleaning
After we explored all the records and fields, we conducted data cleaning by first removing outliers and irrelevant records, and then filling in missing fields.

### 3.1 Exclusions
We removed one outlier which has an extremely large transaction amount ($3102045.53). And we also removed all the transaction types but the purchase transactions (‘P’).

###  3.2 Filling in missing fields
We filled in missing fields in the following procedures.

● Fill in Merchnum

    1. Fill in with mode of Merch description
    
    2. Fill in with “unknown” for adjustment transactions
    
● Fill in Merch state

    1. If the record has a zip, use the state for that zip, if known
    
    2. If in range 00600 – 00799, 00900 – 00999: state = PR (Puerto Rico)
    
    3. Use the mode of the Merchnum or Merch description
    
    4. Fill in with “unknown” for adjustment transactions
    
● Fill in Merch zip

    1. Fill in with the mode of Merchnum or Merch description

    2. Fill in with “unknown” for adjustment transactions
    
● Fill in with ‘unknown’ for all values that are still missing after the above procedures

## 4. Feature Engineering
Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning. Sometimes, the current fields in the raw data cannot offer us sufficient information we need to detect score, so creating new and meaningful variables are important to help us build a more reliable model.

In this section, we will show our methodology for creating new variables. We don’t really care about the relationship between different variables and we only want to create as many meaningful variables as possible. We will cover how to do the feature selection in the next part.

### 4.1 Fraud Types
Card fraud could be introduced from two perspectives, one is the victim and another one is the frauder.

Credit card fraud can happen if someone physically steals the card or virtually hacks the user’s account. From the perspective of victims, there might be large amounts of transactions happening on their cards or accounts in a short time. From the perspective of frauders, they might control multiple accounts and purchase things from the same merch. The number and transactions from one merch could also be important.

### 4.2 Variables Creation
From the introduction of the fraud type above, we could get two conclusions. First, the entities matter. The transactions from one card/merch could be used as a flag of fraud. Second, the measurement methods are also important. The number/ amount/average amount could from different entities offer us information from different perspectives.

Entity

In this case, we create 27 entities from three perspectives: card, merch, location

<img width="470" alt="Screenshot 2023-10-10 at 5 14 18 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/352b3971-98e2-4939-93fe-46e45fd4315f">

<img width="476" alt="Screenshot 2023-10-10 at 5 14 58 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/49b71dab-42c8-4529-bed5-f9f175507d11">

<img width="479" alt="Screenshot 2023-10-10 at 5 15 16 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/7995b5b4-895e-4896-987d-06a1049d58eb">

*State_risk is the fraud risk of each state. The calculation is shown in the following formula. This method could help us smooth the value and avoid overfitting. ( Statistical Smoothing: c = 4, nmid = 20)

<img width="296" alt="Screenshot 2023-10-10 at 5 15 57 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/d0145350-6d9f-4a21-adc4-c877a9a1eb8a">

<img width="888" alt="Screenshot 2023-10-10 at 5 25 38 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/499f18a3-3f66-4744-83fa-d8267ac95cea">

Variable

The variables also covers the following topics: Recency, Frequency, Amount, Velocity Change, Uniqueness, Variability

- Recency (27): The number of days since the last transaction. If it is the first transaction for one card, then use 365 days.
  
- Frequency (189): The number of transactions that happen over the past 0/1/3/7/14/30/60 days across a given entity.
  
- Amount (1512) : The average / median / max / total / (actual/average) / (actual/max) / (actual/median) / (actual/total) of transactions amount happens over past past 0/1/3/7/14/30/60 days across a given entity.
- Velocity Change (216): Velocity change is the number of transactions over the past 0/1 day divided by the average number of transactions over the past 7/14/30/60 days.
  
- Uniqueness (702): Uniqueness is the number of unique entities A across entity B.
  
- Variability (486): Variability is the average of transactions average/median/max change
over the past 0/1/3/7/14/30

As a result, we created 3132 variables after the feature engineering and our next step is to pick up the most meaningful variables from them.

## 5. Feature Selection
The procedurally generated expert variables contain valuable information extracted from the original data; however, they cannot be fed into a machine learning model directly, because the dimensionality is too high. Training a model with such high dimensionality can take an extremely long time, rendering most algorithms computationally intractable. Besides, some models may grow unnecessarily complicated and suffer from overfitting greatly, if we do manage to train them. Thus, feature selection is vital to the success of building machine learning models.

Feature selection is the process of selecting the best subset of relevant features for use in model construction. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

Generally, there are three major categories of feature selection methods: filter, wrapper, and embedded. Filter methods calculate statistics or correlation to rank all features and determine which ones are to be excluded. Wrapper methods, like the name, implies, ‘wrap’ around some machine learning models and use them as predictors to evaluate the importance of each feature. And embedded methods incorporate feature selection within the model training process, intended to reduce the intense computation needed for wrapper methods while retaining decent selection. Here, we use filter and wrapper methods to select our features.

Feature selection techniques are used for several reasons:
● Reduces the complexity of a model and makes it easier to interpret.

● Enables the machine learning algorithm to train faster.

● Avoid the curse of dimensionality.

● Improves the accuracy of a model if the right subset is chosen.

● Reduces overfitting.

### 5.1 Filter
Filter methods select variables regardless of the model. Instead, features are selected on the basis of their scores in various statistical tests for their correlation with the outcome variable. The correlation is a subjective term here. It picks up the intrinsic properties of the features measured via univariate statistics instead of cross-validation performance. Filter methods suppress the least interesting variables. The other variables will be part of a classification or a regression model used to classify or predict data. These methods are particularly effective in computation time and robust to overfitting.

<img width="496" alt="Screenshot 2023-10-10 at 5 18 17 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/91c97107-015a-4980-8f7d-84c8b83fe163">

The following table is generally used to define correlation coefficients.

<img width="485" alt="Screenshot 2023-10-10 at 5 20 41 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/aeb1f7f3-df52-4881-a18a-ade3d4b9b22f">

● Pearson’s Correlation: It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1. Pearson’s correlation is given as:

<img width="129" alt="Screenshot 2023-10-10 at 5 19 03 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b5358c16-f48f-4ece-a08b-b2692bc40bb9">

● LDA: Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.

● ANOVA: ANOVA stands for Analysis of variance. It is similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature. It provides a statistical test of whether the means of several groups are equal or not.

● Chi-Square: It is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.

But one thing that should be kept in mind is that the filter method does not remove multicollinearity. So, we must deal with the multicollinearity of features as well before training models for our data.

Kolmogorov-Smirnov(KS)

KS is a filter method. It is a statistical measure of how well two distributions are separated (goods vs. bads).

● For each candidate variable plot the goods and bads separately

<img width="406" alt="Screenshot 2023-10-25 at 2 02 36 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/c67ee5b9-335b-46aa-9d6e-d5e2110dd8ca">

● The more different the curves the better the variable for separating, and thus the more important the variable.

● KS is a fine and simple measure for how separate are these two curves

<img width="439" alt="Screenshot 2023-10-25 at 2 02 53 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/bd7343fc-a56a-48bd-b30e-c4e43d6d2247">

● Calculate this univariate KS for each variable and throw away the variables with low KS

FDR

Fraud detection rate (FDR) is also used as a filter metrics. Fraud detection rate describes how many frauds we can catch within a certain population, and it is often used as a measurement for model performance, which we are using in this project. In a generic FDR, a machine learning model gives ranking to all records based on the probability of being a fraud, and the records in the given population bin, usually top 3%-5%, are considered as predicted fraud. The fraud detection rate is calculated to be the number of true frauds in the bin, which are caught by the model, divided by the total number of true frauds that exist in the entire dataset. FDR reflects how many frauds can be caught by a model, with a fixed number of predicted positives.

FDR cannot be used as a filter method directly, because it requires a classification model to give a predicted probability list, but a univariate FDR can be used by taking the values of each record as its probability. This is equivalent to a generic classifier which outputs predictions the same as input values. The univariate FDR measures whether frauds tend to cluster at one end of the distribution for one particular feature, and generally perform well in selecting useful features in fraud detection. We apply FDR at a 3% cutoff as a filter metric.

<img width="746" alt="Screenshot 2023-10-25 at 2 03 42 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/a9258682-b62e-4bcd-a7bb-2e2c01f614f3">

### 5.2 Wrapper
The wrapper methods usually select features in a stepwise fashion, where in each step, a machine learning model evaluates the performance of all candidate features exhaustively. It follows a greedy search approach by evaluating all the possible combinations of features against the evaluation criterion. An arbitrary number of top-ranking features can then be selected, and the rest are put into another round of selection until the desired number of features is reached. Compared to filters, wrappers are much more computationally intensive, since a machine learning model is trained for all possible combinations of features at each step, which is why filters are applied before wrapper, to quickly reduce the dimension of the problem. And it usually results in better predictive accuracy than filter methods.

<img width="830" alt="Screenshot 2023-10-25 at 2 04 22 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/a5d4cf2a-2e95-41be-835f-e93c94911a85">

Some common examples of wrapper methods are forward feature selection, backward feature elimination, recursive feature elimination, etc


● Forward Selection: Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.

● Backward Elimination: In backward elimination, we start with all the features and
remove the least significant feature at each iteration which improves the performance of
the model. We repeat this until no improvement is observed in the removal of features.

● Recursive Feature elimination: It is a greedy optimization algorithm that aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.

In our project, forward selection has been used to select variables. Here are results from forwarding selection, simple nonlinear wrapper, and FDR as the measure:
<img width="862" alt="Screenshot 2023-10-25 at 2 05 41 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/94d60ed3-5db3-47a6-b9ed-443cc6d22dd7">

According to the stepwise selection, the number of features to keep is 10. But we decided to include more features to tune the model. So after the wrapper, a total of 20 variables were chosen
to be used for modeling. The final features are listed below.

[List of final 20 variables with their descriptions/definitions]

1. card_zip3_total_7
   
○ Total amount of transactions by the same card & 3-digit zip code combination over
the past 7 days. 

2. Merchnum_max_7
   
○ Maximum amount of transactions by the same merchant over the past 7 days.

4. card_zip_total_14
   
○ Total amount of transactions by the same card & zip code combination over the past 14 days.

6. card_zip_total_60
   
○ Total amount of transactions by the same card & zip code combination over the past
60 days.

8. merch_zip_max_7

○ Maximum amount of transactions by the same merchant & zip code combination over the past 7 days.

10. Card_Merchnum_desc_total_60
    
○ Total amount of transactions by the same card & merchant & merchant description
combination over the past 60 days.

12. zip3_total_0
    
○ Total amount of transactions in the same 3-digit zip code over the past 0 days.

14. card_merch_total_30
    
○ Total amount of transactions by the same card & merchant combination over the past 30 days.

16. card_zip_total_30
    
○ Total amount of transactions by the same card & zip code combination over the past
30 days.

18. card_merch_total_60
    
○ Total amount of transactions by the same card & merchant combination over the past 60 days.

20. Merchnum_desc_total_7
    
○ Total amount of transactions by the same merchant & merchant description
combination over the past 7 days.

12. Merchnum_desc_avg_7
    
○ Average amount of transactions by the same merchant & merchant description combination over the past 7 days.

13. amount_cat
    
14. Merchnum_desc_total_14
    
○ Total amount of transactions by the same merchant & merchant description combination over the past 14 days.

15. Card_Merchnum_desc_total_30
    
○ Total amount of transactions by the same card & merchant & merchant description
combination over the past 30 days. 

16. Card_Merchdesc_total_60

 ○ Total amount of transactions by the same card & merchant description combination over the past 60 days.
 
17. Merchnum_desc_max_7
    
○ Maximum amount of transactions by the same merchant & merchant description
combination over the past 7 days. 

18. merch_zip_total_0
    
○ Total amount of transactions by the same merchant & zip code combination over the past 0 days.

19. zip3_actual/avg_60
    
○ Actual/average amount of transactions in the same 3-digit zip code over the past 60
days.

20. Card_Merchdesc_total_7
    
○ Total amount of transactions by the same card & merchant description combination over the past 7 days.


## 6. Model Algorithms

After selecting the top 20 variables, we started to train various models along with tuning the hyperparameters to find the model that can predict the fraud transactions best. Models we tried in this step include logistic regression, single decision tree, random forest, light gradient boosting machine (LightGBM), extreme gradient boosting (XGBoost), neural network, and cat boost. We will discuss different model techniques as well as their performance in this section.

### 6.1 Logistic Regression

Logistic regression is similar to (multiple) linear regression except that logistic regression is used when the response variable is binary (i.e., assumes only two discrete values). In this card transaction fraud detection project, we set goods to be 0 and bads (fraud transactions) to be 1. We used this model to predict if the particular transaction is a fraud or not.

A logistic regression model is:

<img width="304" alt="Screenshot 2023-10-25 at 2 12 09 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/667782bd-1315-4e9b-aaa0-fd215f8316ac">

<img width="890" alt="Screenshot 2023-10-25 at 2 12 32 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/500864dc-536d-4c2f-9e45-5b12f299de01">

Hyperparameter tuned:

● penalty: three norms of penalty -- ‘l1’, ‘l2’, ‘none’

1. none: no penalty is added
   
2. l2: add an L2 penalty term and it is the default choice;
   
3. l1: add an L1 penalty term;
   
● C: Inverse of regularization strength.

● solver: Algorithm to use in the optimization problem.

● max_iter: Maximum number of iterations taken for the solvers to converge.

Performance:

<img width="863" alt="Screenshot 2023-10-25 at 2 13 39 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/1d319bbe-b4d2-4a47-af94-45c93caf4872">

### 6.2 Decision Tree

Decision tree classifier is a tree-based model which starts at the root node and ends with a decision at the leaves (terminal nodes). The decision node at each level is determined by attribute selection based on the Gini impurity or the Shannon information gain. At each terminal node, a class label is assigned (fraud or not). We tuned the model by adjusting split criteria, the maximum depth of the tree, the minimum number of samples required to be at a leaf node, and other parameters.

<img width="847" alt="Screenshot 2023-10-25 at 2 14 20 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/95ab1eeb-4f33-490f-a109-a9425b88594c">

Hyperparameter tuned:

● criterion: The function to measure the quality of a split.

● splitter: The strategy used to choose the split at each node.

● max_depth: The maximum depth of the tree.

● min_samples_split: The minimum number of samples required to split an internal node.

● min_samples_leaf: The minimum number of samples required to be at a leaf node.

● max_features: The number of features to consider when looking for the best split.

Performance:

<img width="877" alt="Screenshot 2023-10-25 at 2 15 05 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/870e1ffc-0698-43b6-abc9-99e74aad0241">

### 6.3 Gradient Boosting

Boosted Trees is another ensemble method based on a decision tree. Unlike random forests, which vote the results from strong and deep trees, Boosted Trees method builds a series of many weak trees and integrates them. Boosting is a way of training a series of weak models to result in a strong model. Each weak model is trained to predict the residual error of the current sum. One new weak learner is added at a time and existing weak learners in the model are frozen and left unchanged. Gradient boosting is considered a gradient descent algorithm. Gradient descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of gradient descent is to tweak parameters iteratively in order to minimize a cost function. We tuned the model by adjusting the number of boosting stages, and the maximum depth of the individual regression estimators and other parameters.

The gradient boosting uses an ensemble of weak learners such as decision trees to predict the outcome variable. Each subsequent tree in the series tries to capture the residual errors of the previous tree. We used XGBoost, which is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

<img width="826" alt="Screenshot 2023-10-25 at 2 16 00 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/2cfad018-ad6b-4a7b-97a2-557d9c4b528c">

Hyperparameter tuned:

● max_depth: The maximum depth of the individual regression estimators.

● learning_ rate: Learning rate shrinks the contribution of each tree.

● min_samples_split: The minimum number of samples required to split an internal node.

● min_samples_leaf: The minimum number of samples required to be at a leaf node.

● n_estimators: The number of boosting stages to perform.

<img width="880" alt="Screenshot 2023-10-25 at 2 16 34 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/98c98ecf-8fed-4e97-98f0-44a537831489">

### 6.4 Random Forest

Random Forest is another tree-based supervised machine learning model. It is called a “forest” because it grows a forest of decision trees. The model uses only a randomly-chosen subset of variables or records for each tree and/or for each split iteration within a tree. If the goal of the model is to get a regression, the model will output a result by averaging the value from each tree; if the goal is to classify, like what we do in this case, the model combines the results by voting.

<img width="888" alt="Screenshot 2023-10-25 at 2 17 01 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/7557c3e0-2d4c-4a4b-9fd5-c7118f4eb5de">

Hyperparameter tuned:

● n_estimators: The number of trees in the forest.

● max_depth: The maximum depth of the tree.

● min_samples_split: The minimum number of samples required to split an internal node.

● min_samples_leaf: The minimum number of samples required to be at a leaf node.

● max_features: The number of features to consider when looking for the best split.

● bootstrap: Whether bootstrap samples are used when building trees.

Performance:

<img width="859" alt="Screenshot 2023-10-25 at 2 17 47 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/5b7e9614-05e1-4a71-8eec-aa7227919636">

### 6.5 Light GBM

LightGBM is a gradient boosting framework based on decision trees to increase the efficiency of the model and reduce memory usage. Light GBM grows trees vertically while another algorithm grows trees horizontally meaning that Light GBM grows trees leaf-wise while another algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, a Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

<img width="823" alt="Screenshot 2023-10-25 at 2 18 24 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/7ce7aee9-d02c-469b-bef7-829bdffdd5b3">

<img width="838" alt="Screenshot 2023-10-25 at 2 18 42 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/50e2a35c-303b-411f-a8fc-997b14b30c08">

Hyperparameter tuned:

● n_estimators: The number of trees in the forest.

● max_depth: The maximum depth of the tree.

● n_leaves: The number of leaves of the tree.

● learning_Rate: The learning rate of the gradient algorithm.

Performance:

<img width="869" alt="Screenshot 2023-10-25 at 2 19 12 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/3af9904e-e8bd-41a3-a70a-9febb3126aa5">

### 6.6 Neural Network

Neural Network is a Deep Learning technic to build a model according to training data to predict unseen data using many layers consisting of neurons. This is similar to other Machine Learning algorithms, except for the use of multiple layers. The use of multiple layers is what makes it Deep Learning. It works similarly to the human brain’s neural network. A “neuron” in a neural network is a mathematical function that collects and classifies information according to a specific architecture. The network bears a strong resemblance to statistical methods such as curve fitting and regression analysis.

A neural network contains layers of interconnected nodes. Each node is known as a perceptron and is similar to multiple linear regression. The perceptron feeds the signal produced by a multiple linear regression into an activation function that may be nonlinear.

Some of the key characteristics of Neural Networks are:

● Neural networks are a series of algorithms that mimic the operations of an animal brain to recognize relationships between vast amounts of data.

● As such, they tend to resemble the connections of neurons and synapses found in the brain.

● They are used in a variety of applications in financial services, from forecasting and marketing research to fraud detection and risk assessment.

● Neural networks with several process layers are known as "deep" networks and are used for deep learning algorithms.

<img width="824" alt="Screenshot 2023-10-25 at 2 20 13 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/ab147b1b-a28e-4929-b678-a14abd552c2b">

Hyperparameter tuned:

● max_iter: Maximum number of iterations.

● hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden
layer.

● activation: Activation function for the hidden layer.

● solver: The solver for weight optimization.

● learning_rate: Learning rate schedule for weight updates.

Performance:

<img width="864" alt="Screenshot 2023-10-25 at 2 20 57 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/6e519ad9-2479-4df3-8109-419abda58a82">

### 6.7 CatBoost

CatBoost is an open-source machine learning(gradient boosting) algorithm, with its name coined from “Category” and “Boosting.”

Key features of CatBoost:

● Symmetric trees: CatBoost builds symmetric (balanced) trees, unlike XGBoost and LightGBM. In every step, leaves from the previous tree are split using the same condition. The feature-split pair that accounts for the lowest loss is selected and used for all the level’s nodes. This balanced tree architecture aids in efficient CPU implementation, decreases prediction time, makes swift model appliers, and controls overfitting as the structure serves as regularization.

<img width="743" alt="Screenshot 2023-10-25 at 2 21 54 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/4286ef0e-a99d-40c6-8ab9-8df0329ae668">

● Ordered boosting: Classic boosting algorithms are prone to overfitting on small/noisy datasets due to a problem known as prediction shift. When calculating the gradient estimate of a data instance, these algorithms use the same data instances that the model was built with, thus having no chance of experiencing unseen data. CatBoost, on the other hand, uses the concept of ordered boosting, a permutation-driven approach to train the model on a subset of data while calculating residuals on another subset, thus preventing target leakage and overfitting.

● Native feature support: CatBoost supports all kinds of features be it numeric, categorical, or text and saves time and effort of preprocessing.

Hyperparameter tuned:

● learning_rate: The learning rate. Used for reducing the gradient step.

● iterations: The maximum number of trees that can be built when solving machine learning problems.

● depth: Depth of the tree.

● l2_leaf_reg: Coefficient at the L2 regularization term of the cost function. Any positive
value is allowed.

Performance:

<img width="831" alt="Screenshot 2023-10-25 at 2 23 33 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/850cd736-f9df-4e1b-a351-1ee62d74f79a">

## 7. Results

The final model we chose to implement is a neural network model with the following hyperparameters:
MLPClassifier(hidden_layer_sizes=(200,), alpha=.005, solver='adam', activation='logistic', max_iter=200, learning_rate='adaptive', learning_rate_init=.01)

We chose to use the top 10 variables that we got from the feature selection process to training the model as it gives us a well-performed result and signals relatively small overfitting, compared to using all of the top 20 variables.

We did a 0.7/0.3 train test split and reserved the records from the last two months as out-of-time validation datasets. The final performance summary of our model on the training, testing, and validation datasets are as shown below:

<img width="857" alt="Screenshot 2023-10-25 at 2 24 24 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/157c2ab5-9c2b-41d6-8671-19c8db91bae3">

<img width="858" alt="Screenshot 2023-10-25 at 2 24 40 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b7d524d8-839c-4127-a3ce-1422d1ed4636">

<img width="852" alt="Screenshot 2023-10-25 at 2 24 56 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/4df20a14-4bea-41d8-8009-11aa987ce2a1">

8. Conclusion

In summary, we first explore the original data, knowing the quality of it and make a exploratory data analysis to have a better understanding of the data we have. After discussing the possible reason of card fraud, we create 27 entities and 3132 new variables based on the amount, frequency and recency of the transactions and etc. Next, we use Kolmogorov–Smirnov test to keep the variables that have high KS-score and then use random forest forward selection wrapper to do a further selection. As a result, we successfully reduce the number of variables into 20.

As for modeling, we have tried 7 types of machine learning algorithms including logistic regression, decision tree, gradient boosting, random forest, light GBM, neural network and catboost. After comparing the performance in OOT data and the risk of overfitting, the neural network model with the hyperparameters of { hidden_layer_sizes=(200,), alpha=.005, solver='adam', activation='logistic', max_iter=200, learning_rate='adaptive', learning_rate_init=.01} and 10 variables performed the best. The final model has an average 54.2% FDR at 3% in the out-of-time validation dataset and doesn’t face the risk of overfitting.

Detecting fraud is an challenging but meaningful work. If we have the chance to improve the work we have done, we might seek some expert advice and build some expert variables. In addition, if we could have a lager dataset, we might have a better result.

## 9. Appendix

### 9.1 Appendix 1 – Data Quality Report (DQR)

Data Description:

● The data provided is “Actual Credit Card Purchases” from the US government. It provides credit card transaction identifying information including data like card number, transaction date, the amount of transaction, whether the transaction is fraudulent, etc.

● Where it came from: This data is actual credit card purchases from a US government organization.

● Time period it covers: This is a 12-month data covering the period from 2006-01-1 to 2006-12-31. And the data is sorted chronologically by “Recnum”.

● Number of fields: There are 10 columns/fields. The “Fraud” field is a dependent variable with a label indicating fraud (0 represents “no fraud”, and 1 represents “fraud”).

● Number of records: There are 96,753 records. Only 1,059 of them are frauds records. 

Field Summary Tables

<img width="908" alt="Screenshot 2023-10-25 at 2 26 40 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/e10b7875-46eb-4411-8fd9-241ef96677a9">

Categorical Fields Summary Table

<img width="638" alt="Screenshot 2023-10-25 at 2 32 50 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/8dc0123b-5d08-42f5-b1be-5c4095dee446">

Distribution of Fields

1. Recnum
   
● This field is a unique identifier for each record, essentially a serial number.

● All the records are sorted chronologically by “Recnum”, which can be regarded as time
order.

● Each row here has a unique non-null value.

2. Cardnum
● This field is a categorical variable classifying the credit card number of each transaction.

● There are 1,645 card numbers having fraud records, and the card number 5,142,148,452
defrauded the most times.

<img width="838" alt="Screenshot 2023-10-25 at 2 34 01 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/cd4bbf7d-b4c2-437e-b46f-f05f5667386e">

3. Date
   
● This field represents when the credit card transaction happened.

● It covers a period of 12 months from January 1st, 2006 to December 31st, 2006, which
means all the transactions are happened and evaluated in 2006.

● And all the records are sorted chronologically according to the transaction date.

● February 28th, 2006 saw the highest number of credit card transactions.

● From the weekly and monthly transaction plots we can see that there is a sharp drop after
September. It is because the U.S. government has a fiscal year that starts on October 1st. So the end of September is also the end of the fiscal year for the government. It's the fact that this is a government purchase card that makes it different from consumers’ purchases. The government gets its budget to reset on October 1st. So it might be conservative and spend little at the beginning of a new fiscal year.

<img width="825" alt="Screenshot 2023-10-25 at 2 34 43 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b70e65a4-2e51-4a2c-863d-6292e076b588">

<img width="811" alt="Screenshot 2023-10-25 at 2 34 59 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/47028b58-7820-45f6-ac9e-45cfdaea1530">

<img width="849" alt="Screenshot 2023-10-25 at 2 35 12 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/4c081d6f-7f02-4de4-a926-9d24fcabc1bc">

<img width="841" alt="Screenshot 2023-10-25 at 2 35 27 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b815e34f-46ac-4e17-802b-5b3cca67e0ed">

4. Merchnum

● This is a categorical field containing the merchant number of each transaction.

<img width="842" alt="Screenshot 2023-10-25 at 2 35 58 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/5bccdc98-53e9-46ce-b92d-7a9c475bad10">

5. Merch description
   
● This field contains the description of each merchant.

<img width="859" alt="Screenshot 2023-10-25 at 2 36 45 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/bcb9c76b-8850-47f8-a387-78bf482835b0">

6. Merch state
   
● Merch state is a categorical variable representing the state of the merchant.

● Most of the merchants of these transactions are in Tennessee.

● There are 227 unique values for this field. Besides the abbreviation for 50 states in the
US, there are many three-digit values, I infer that it’s the three-digit zip code of specific regions. There also exist some areas outside the US, like BC, QC and ON, etc.

<img width="835" alt="Screenshot 2023-10-25 at 2 37 40 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/252ce9fb-ef0d-482c-be3a-229818eb585b">

7. Merch zip
   
● This field contains the zip code of each merchant.

<img width="803" alt="Screenshot 2023-10-25 at 2 38 04 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/ac0fafe3-2f95-4a9a-bb64-d9de42050a8a">

8. Transtype

● This field is a categorical variable that indicates the transaction type.

<img width="820" alt="Screenshot 2023-10-25 at 2 38 36 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/d5861354-344f-4967-881a-557ac79a841f">


9. Amount
    
● This field is a numeric variable that indicates the amount of each transaction.

● The distribution is right-skewed. Most of the transaction amounts are under $2500. There
are fewer large-amount transactions than small-amount transactions.

● From the distribution of the amount of fraud and no fraud records, we can see that generally there are more fraudulent transactions than non-fraudulent transactions when the transaction amount ranges from $500 to $13000.

● In the range between 13000 and 23000, overall, there are more non-fraudulent
transactions than fraudulent transactions.

● We can infer that fraud often happens on transactions with extremely small amounts and
transactions with amounts between 13000 and 23000.

<img width="763" alt="Screenshot 2023-10-25 at 2 39 19 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/ba13b151-8b90-4c45-a3ee-5af1b9c107f6">

<img width="698" alt="Screenshot 2023-10-25 at 2 39 30 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/b28ddbe5-6267-4131-977d-365ad7fcfe14">

10. Fraud
    
● This field is a categorical variable describing whether the transaction had fraud or not, where 0 represents no fraud and 1 represents fraud.

● There are very few frauds in these transactions.

<img width="853" alt="Screenshot 2023-10-25 at 2 39 54 PM" src="https://github.com/jennyliyiyuan/Card-Transaction-Fraud-Detection/assets/133256378/9c73408d-a4e2-49ac-b76b-eb0de4c4176c">
