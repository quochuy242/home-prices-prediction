import numpy as np
import pandas as pd

# Read train and test data
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

# Combine train and test data for data cleaning
y = train.SalePrice
train.drop("SalePrice", axis=1)

train["train_test"] = 1
test["train_test"] = 0

X = pd.concat([train, test])

# Create categorical columns
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
# Create numeric columns
numerical_cols = train.select_dtypes(exclude=["object"]).columns.tolist()

"""
Split features further: continuous, discrete, ordinal, nominal, alphanumerical and binary

Continuous features: 'LotFrontage', 'LotArea', 'MasVnrArea', '
    BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
    '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'

Discrete features: 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd', 
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold', 'BsmtFullBath', 
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr'

Ordinal features: 'OverallQual', 'OverallCond', 'LotShape', 'ExterQual', 'ExterCond', 
    'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 
    'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 
    'PavedDrive'(Ordinal or nominal?), 'PoolQC', 'Fence'(Ordinal or nominal?)

Nominal features: 'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 
    'LandSlope', 'Neighborhood', 'Condition1','Condition2', 'RoofStyle', 'RoofMatl', 
    'Exterior1st','Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 
    'Functional', 'GarageType', 'MiscFeature', 'SaleType','SaleCondition', 'MSSubClass'

Alphanumerical features: 'LotConfig','BldgType','HouseStyle'

Binary features: CentralAir

Temp features: train_test

N.B. MSSubClass appears to be a discrete features but is infact a nominal one


"""

train["MSSubClass"] = train["MSSubClass"].astype("str")
test["MSSubClass"] = test["MSSubClass"].astype("str")
X["MSSubClass"] = X["MSSubClass"].astype("str")
numerical_cols.remove("MSSubClass")
categorical_cols.append("MSSubClass")

# Continuous: lien tuc
cont_features = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
]

# Discrete: roi rac
disc_features = [
    "MSSubClass",
    "YearBuilt",
    "YearRemodAdd",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "MoSold",
    "YrSold",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
]

# Ordinal:
ord_features = [
    "OverallQual",
    "OverallCond",
    "LotShape",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
]

# Nominal
nom_features = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "Utilities",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "Functional",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]

# Alphanumerical
alpnum_features = ["LotConfig", "BldgType", "HouseStyle"]
# Binary
bin_features = ["CentralAir"]

temp_features = ["train_test"]
