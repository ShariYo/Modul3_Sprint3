
# Project: Spaceship Titanic

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!  
To help rescue crews and retrieve the lost passengers, we are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship's computer system.


## Data source

Given data source can be found at Kaggle with the link provided below:

  **Spaceship Titanic's Transported Passengers Prediction Data**  

  https://www.kaggle.com/competitions/spaceship-titanic/data



## Summary
 
"Transported" prediction data was analyzed and results presented. There were 13 different features and 1 target - Transported. Features consisted of object (categorical) and int/float numeric dtypes. Some of object dtypes were converted to numeric. Herein EDA insights:
- Given data had some NaN values almost in all feature columns.
- All float dtypes were converted to int.
- Age data distribution is skewed to the right with the most data points of 20-30 age passengers. Interestingly, 0-10 were more transported than not.
- Age feature was converted to have age interval classes.
- Luxurity features (room service, food court, shopping mall, spa, vrdeck) most of the data are near 0. With increasing money spent, amount of data points drops dramatically. Data is strongly skewed to the right with a long tail. To mitigate these problems, luxury features were transformed to logarithmics.
- Travellers who travelled to Europa were Transformed two times more and almost 5 times more who were on cryosleep.
- VIP status does not increase number of transported people.
- Feature "Cabin" was splitted into 3 new features: cabindeck, cabinnum, and cabinside. This action should increase model's performance.
- Categorical features degree of depency (DOD) on target did show higher dependency of passengerid, name and cryosleep features.
- To check multicolinearity VIF model was used. No strong correlation between features was observed.
- Target itself (Transported) has very balanced data which positively impacts ML model prediction quality.
**ML prediction**  
For ML prediction 5 models were chosen as candidates: KNN, Logistic Regression, SVC, RandomForestClassifier, XGBoost and executed with datasets as a raw models. The best performed model (XGBClassifier) has been choosed for further analysis and tuning.
- For prediction preprocessor (ColumnTransformer) was used. Everything fitted in pipeline.
- To tune XGBClassifier, these hyperparameters were used: max_depth, n_estimators, learning_rate, min_child_weight, colsample_bytree.
- The results for XGBClassifier indicate strong overall performance with an accuracy of 0.81. Model's precision and recall are well-balanced for both classes with "False" achieving 0.81, 0.79 for recall and F1-score of 0.80. Results for "True" goes accordingly 0.80, 0.82, 0.81. The macro average and weighted average F1-scores are both 0.80 confirming that model performs consistently across both classes without favoring one over other These results highlight the classifier's strong ability to minimize both false positives and false negatives, making it suitable for tasks requiring balanced classification accuracy.

**What could have been done more**: 
- H0 hypothesis with different features could have been analysed;  
- Visualize summarized ML results to look for more insights. 
- Try SHAP, ELI5, LIME to find which features impacts more highly. 
- In functions_sandbox.py edit some of the functions by adding more flexibility and durability.

