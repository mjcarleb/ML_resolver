import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data_dir = "../DataGeneration/data/"
file_name = "training_trades"
df = pd.read_parquet(path=f"{data_dir}{file_name}")

train, test = train_test_split(df,test_size=0.15, random_state=42)

########################################
#     PREPARE DATA
########################################
features = ["market", "source_system"]
label = "resolver_label"

feature_enc = OneHotEncoder()
feature_fit = feature_enc.fit(X=train[features])
train_X = feature_enc.transform(X=train[features])

label_enc = OneHotEncoder()
label_fit = label_enc.fit(X=train[label].values.reshape(-1, 1))
train_y = label_enc.transform(X=train[label].values.reshape(-1, 1))

########################################
#     TRAIN MODEL
########################################
clf = DecisionTreeClassifier(random_state=42)

model = clf.fit(X=train_X.toarray(), y=train_y.toarray())
a=3
