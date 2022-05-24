import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn import tree

data_dir = "../DataGeneration/data/"
file_name = "training_trades"
df = pd.read_parquet(path=f"{data_dir}{file_name}")

train, val = train_test_split(df,test_size=0.15, random_state=42)

########################################
#     PREPARE DATA
########################################

# ohe features
ohe_features = ["market", "source_system", "account", "sanctioned_security"]
feature_enc = OrdinalEncoder()  #OneHotEncoder(sparse=False)
feature_enc.fit(X=train[ohe_features])
ohe_tr = feature_enc.transform(X=train[ohe_features])
ohe_val = feature_enc.transform(X=val[ohe_features])

# now add the float feature for amount
amt_USD_tr = train["amount_USD"].to_numpy(dtype=float).reshape((ohe_tr.shape[0], -1))
amt_USD_val = val["amount_USD"].to_numpy(dtype=float).reshape((ohe_val.shape[0], -1))
X_tr = np.concatenate((ohe_tr, amt_USD_tr), axis=1)
X_val = np.concatenate((ohe_val, amt_USD_val), axis=1)
a=3

# label
label = "resolver_label"
label_enc = LabelEncoder() # OneHotEncoder(sparse=False)
label_enc.fit(train[label])  #label_enc.fit(X=train[label].values.reshape(-1, 1))
y_tr = label_enc.transform(train[label]) #label_enc.transform(X=train[label].values.reshape(-1, 1))
y_val = label_enc.transform(val[label]) #label_enc.transform(X=val[label].values.reshape(-1, 1))
a=3

########################################
#     TRAIN MODEL
########################################
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X=X_tr, y=y_tr)

########################################
#     EVALUATE THE MODEL
########################################
predictions = dt_model.predict(X_val)
accuracy = accuracy_score(y_true=y_val, y_pred=predictions)
print(f"Accuracy with features = {ohe_features} = {accuracy :0.2%}")

########################################
#     PERSIST MODELS
########################################
model_dir = "models/"

file_name = "feature_enc_model.serial"
with open(f"{model_dir}{file_name}", "wb") as f:
    pickle.dump(feature_enc, f)

file_name = "label_enc_model.serial"
with open(f"{model_dir}{file_name}", "wb") as f:
    pickle.dump(label_enc, f)

file_name = "dt_model.serial"
with open(f"{model_dir}{file_name}", "wb") as f:
    pickle.dump(dt_model, f)


##################################################
#     RETRIEVE AND CONFIRM SERIALIZATION WORKED
##################################################
with open(f"{model_dir}{file_name}", "rb") as f:
    dt_model = pickle.load(f)

predictions = dt_model.predict(X_val)
accuracy = accuracy_score(y_true=y_val, y_pred=predictions)
print(f"After ser/deser, accuracy with features of {ohe_features} anda amount_USD:  {accuracy :0.2%}")


##################################################
#          VIZ
##################################################
fig = plt.figure(figsize=(12,3), dpi=2000)
tree.plot_tree(dt_model, feature_names=["market", "source_system", "account", "sanctioned_security", "amt_USd"],
                   class_names=["compliance", "peter", "ema", "mary", "bill", "sue", "suzy", "phil", "richard", "abdul", "kevin", "darien", "daren"], filled=True)

fig.savefig("tree.png")