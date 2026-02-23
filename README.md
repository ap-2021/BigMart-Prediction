# BigMart-Prediction
sales prediction problem

1. Problem Understanding
The objective was to predict Item_Outlet_Sales for products across different outlets. This is a supervised regression problem using structured tabular data comprising:
•	Item-level attributes (MRP, type, visibility, fat content, weight)
•	Outlet-level attributes (size, type, location, establishment year)
•	Historical sales
The evaluation metric was Root Mean Squared Error (RMSE)
2. Exploratory Data Analysis (EDA)
Using Pandas profiling and distribution plots, I conducted structured EDA to understand data behavior and quality.( refer to file : EDA_bigmart_profile)
Key observations:
•	Sales distribution was right-skewed.
•	Item_MRP showed strong correlation with sales.
•	Outlet-level variables (size, type, establishment year) were correlated.
•	Missing/zero values existed in:
o	Item_Weight
o	Outlet_Size
o	Item_Visibility
•	Item_Fat_Content required standardization.
•	Certain Item_Type categories needed grouping due to skewed frequency distribution.
These findings directly guided feature engineering and modeling decisions.
3. Data Preprocessing & Feature Engineering
Data Cleaning
•	Standardized categorical inconsistencies (e.g., Item_Fat_Content).
•	Imputed:
o	Item_Weight using item-level mean.
o	Outlet_Size using outlet-type grouping.
•	Replaced zero Item_Visibility with item-level average visibility.
Feature Engineering
•	Created Outlet_Age = Current_Year - Outlet_Establishment_Year.
•	Built Visibility_Ratio = Item_Visibility / Average_Item_Visibility.
•	Extracted Item_Category from Item_Identifier.
•	Created outlet-level aggregate features:
o	Outlet_Average_Sales
o	Outlet_Item_Count
•	Applied target encoding to Outlet_Identifier to capture store-level sales behavior.
•	Explored transformations and interactions:
o	Polynomial terms of Item_MRP
o	Log of MRP
o	Log transformation of target
o	Interaction between Item_MRP and outlet-level aggregates
These features helped capture strong price-driven and store-level effects.
4. Model Experimentation Strategy
I experimented with both bagging and boosting approaches:
•	Linear Regression (baseline)
•	Random Forest
•	LightGBM
•	XGBoost
•	Ensemble of Random Forest + LightGBM
Observations:
•	Tree-based models significantly outperformed linear models.
•	Simple weighted averaging provided modest improvements.
•	Meta-model stacking did not add value due to high correlation between base models and risk of overfitting.
•	Final chosen model: Random Forest (best bias-variance tradeoff in CV).
5. Validation & Hyperparameter Optimization
•	Used 5-Fold Cross-Validation for robust performance estimation.
•	Hyperparameters tuned using Optuna with 5-fold CV, optimizing:
The optimization objective was minimizing mean CV RMSE.
6. Error Analysis & Iterative Refinement
Performed structured error analysis to:
•	Identify high-error sales zones.
•	Detect feature instability across folds.
•	Evaluate necessity of transformations and interaction terms.
This analysis led to selective addition of price transformations and outlet-level encoding while avoiding excessive feature engineering.
7. Key Insights
•	Item_MRP is the strongest predictor.
•	Outlet-level behavior significantly impacts sales.
•	Target encoding of outlet improves predictive stability.
•	Over-engineering interactions does not necessarily reduce RMSE.
•	Controlled regularization and proper CV strategy are critical for stable performance.

