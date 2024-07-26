# Bangalore-Real-Estate-Price-Prediction
### Abstract:
The Bangalore Real Estate Price Prediction project aimed to create a machine learning model to accurately predict property prices based on features like location, total square footage, number of bathrooms, and BHK. Data from Kaggle was cleaned and processed, including handling missing values, transforming columns, and performing dimensionality reduction by grouping less frequent locations. Outliers were removed based on business logic and statistical deviation. Feature engineering included creating a price_per_sqft column and converting categorical data into dummy variables. A Linear Regression model was trained and validated using cross-validation, achieving an R² score of approximately 0.84. The model's performance was further tested using cross-validation techniques, ensuring its robustness and accuracy.

### Objective:
To develop a robust machine learning model capable of accurately predicting property prices in Bangalore. The objective encompasses the exploration, preprocessing, feature engineering, and training phases to create a predictive model that can assist homebuyers, real estate professionals, and stakeholders in making informed decisions regarding property investments in the Bangalore real estate market. The ultimate goal is to achieve high predictive performance, generalizability, and interpretability, thereby providing valuable insights into the dynamic and heterogeneous nature of property pricing trends in Bangalore.

### Data Source:
Dataset used here in this project has been taken from Kaggle.

### Data Exploration:
![image](https://github.com/user-attachments/assets/3f671f06-bf5e-4852-a11f-cd314fff508a)
![image](https://github.com/user-attachments/assets/24f6ed18-f8c0-4f0e-8784-01ecf05efac5)
![image](https://github.com/user-attachments/assets/39518991-3204-4a88-a274-d5bb1f086347)

So, 13320 rowes and 9 columns were there in the dataset.

### Data Cleaning:
![image](https://github.com/user-attachments/assets/cf07da34-656a-4bfb-935e-51094a81e2f2)
* Irrelevant columns such as Availability, Balcony, Area_type, and Society were dropped from the dataset.

![image](https://github.com/user-attachments/assets/4f935d45-79e7-4f8e-b2ba-a03b9c934d27)
* 90 rows with null values were dropped as they constituted a small fraction of the dataset.

![image](https://github.com/user-attachments/assets/2275d748-2e9e-4db9-b64c-75b92e708bfe)
![image](https://github.com/user-attachments/assets/14fd5e02-739b-4e12-a67e-baa154b03c91)
* The Size column was transformed to extract the number of BHKs, creating a new column BHK.

![image](https://github.com/user-attachments/assets/0e164370-7f5d-4b65-9ced-4b5d34af921c)
* The total_sqft column contained mixed formats (integers, ranges, and units). These were standardized by replacing ranges with their mean values.

### Dimensionality Reduction:

![image](https://github.com/user-attachments/assets/82a442fd-5ac3-4cf2-bc2b-20f6dc4bd6e4)
![image](https://github.com/user-attachments/assets/f9c1c778-e993-4d23-bd1c-eccceaddf867)
![image](https://github.com/user-attachments/assets/7e94b4c3-a6e5-456b-9271-4f6f4afd1e76)
* Locations with fewer than 10 data points were grouped into an "others" category, reducing the number of unique locations from 1052 to 242.

### Outlier Removal:

**1. Based on Business Logic:**
Transactions where total_sqft per BHK was less than 300 were dropped.

**2.Based on Statistical Deviation:**
Only records with price_per_sqft within one standard deviation of the mean were retained.

**3.Consistency Checks:**
Transactions where 2 BHK flats cost more than 3 BHK flats in the same location and area were removed.
A threshold was set where the number of bathrooms could not exceed the number of BHKs by more than 2.

### Final Data Preparation:
![image](https://github.com/user-attachments/assets/e463a5cf-3958-4a50-a3a4-b5ac2aa13863)
![image](https://github.com/user-attachments/assets/3d2f9f2a-e260-4f8a-8cb8-087e9a46b1d8)

* Dummy variables were created for the categorical location feature, with one category dropped to avoid the dummy variable trap.
* The dummy variables were concatenated with the original dataframe, and the location column was dropped.

### Model Building:
**1. Data Splitting:**
![image](https://github.com/user-attachments/assets/6a5bcb73-9745-4e13-8404-543d8c05b13c)

* The dataset was split into training and testing sets using train_test_split().

**2. Model Training:**
![image](https://github.com/user-attachments/assets/8b682244-5b8b-4388-b68e-8f0e7cf415a6)

* A Linear Regression model was trained using lr.fit(X_train, Y_train).
* The model's performance was evaluated using R² score on the test set, yielding approximately 0.84.

**3. Model Validation:**
![image](https://github.com/user-attachments/assets/48babc9a-299e-4485-936d-9a4a11900dff)

* Cross-validation was performed using ShuffleSplit with 5 splits, validating the model with cross_val_score().

### Conclusion:
The project successfully built a Linear Regression model to predict Bangalore real estate prices with an R² score of over 0.80. The model was validated using cross-validation, ensuring its robustness. The comprehensive data cleaning, outlier removal, and feature engineering steps were crucial in achieving this performance.




















