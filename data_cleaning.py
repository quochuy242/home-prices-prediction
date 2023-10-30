import numpy as np
import pandas as pd

"""
In the data_visualisation.ipynb file, we defined useless features:

1)No Correlation with SalePrice: 'YrSold', 'MoSold'

2)Quasi-constant Categorical: 'Street', 
    'Utilities', 'Condition2', 'RoofMatl', 'Heating'

3)Quasi-constant Numerical: 'LowQualFinSF', 
    'KitchenAbvGr', '3SsnPorch', 'PoolArea', 'MiscVal'
"""
# Import data_analysis.py
import data_analysis as da

# Removing useless features
da.train = da.train.drop(
    [
        "MoSold",
        "Street",
        "Utilities",
        "Condition2",
        "RoofMatl",
        "Heating",
        "LowQualFinSF",
        "KitchenAbvGr",
        "MiscVal",
        "PoolArea",
    ],
    axis=1,
)
da.test = da.test.drop(
    [
        "MoSold",
        "Street",
        "Utilities",
        "Condition2",
        "RoofMatl",
        "Heating",
        "LowQualFinSF",
        "KitchenAbvGr",
        "MiscVal",
        "PoolArea",
    ],
    axis=1,
)

# Keep'YrSold' for time being as may be useful in feature engineering.

# Removing outliers
"""We use IQR"""
"""Calculate the upper and lower limits"""
Q1 = da.train[da.numerical_cols].quantile(0.25)
Q3 = da.train[da.numerical_cols].quantile(0.75)

IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

upper_index = np.where(da.train[da.numerical_cols] >= upper)[0]
lower_index = np.where(da.train[da.numerical_cols] <= lower)[0]

da.train = da.train.drop(index=upper_index, inplace=True)
da.train = da.train.drop(index=lower_index, inplace=True)

# Filling NA values
"""Firstly, we combine train and test data again to account 
for the outliers we removed"""

X = pd.concat([da.train, da.test])

na_df = pd.DataFrame({"Features": X.isnull().sum(axis=0) / len(X)}).sort_values(
    by="Features", ascending=False
)
na_df = na_df.loc[na_df["Features"] != 0]

none_cols = [
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "GarageType",
    "BsmtCond",
    "BsmtExposure",
    "BsmtQual",
    "BsmtFinType2",
    "BsmtFinType1",
    "MasVnrType",
]

for col in none_cols:
    X[col] = X[col].fillna("None")

zero_cols = [
    "MasVnrArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "GarageYrBlt",
    "GarageCars",
    "TotalBsmtSF",
    "BsmtUnfSF",
    "GarageArea",
]
for col in zero_cols:
    X[col] = X[col].fillna(0)


mode_cols = ["KitchenQual", "Electrical", "SaleType", "Exterior1s", "Exterior2nd"]
for col in mode_cols:
    X[col] = X.groupby("Neighborhood")[col].transform(lambda x: x.fillna(x.mode()[0]))

X["MSZoning"] = X.groupby("MSSubClass")["MSZoning"].transform(
    lambda x: x.fillna(x.mode()[0])
)

X["Functional"] = X["Functional"].fillna("Typ")

# Mapping Categorical Features
X["CentralAir"] = X["CentralAir"].map({"N": 0, "Y": 1})
X["LotShape"] = X["LotShape"].map({"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4})
X["LandSlope"] = X["LandSlope"].map({"Gtl": 1, "Mod": 2, "Sev": 3})
QualCondMap = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
for col in [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "GarageQual",
    "GarageCond",
    "KitchenQual",
    "HeatingQC",
    "FireplaceQu",
    "PoolQC",
]:
    X[col] = X[col].map(QualCondMap)
X["BsmtExposure"] = X["BsmtExposure"].map(
    {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
)
X["GarageFinish"] = X["GarageFinish"].map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3})
X["PavedDrive"] = X["PavedDrive"].map({"N": 1, "P": 2, "Y": 3})
X["Fence"] = X["Fence"].map({"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0})
