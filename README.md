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
