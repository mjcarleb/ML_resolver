import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

data_dir = "../DataGeneration/data/"
file_name = "training_trades"
df = pd.read_parquet(path=f"{data_dir}{file_name}")

train, val = train_test_split(df,test_size=0.15, random_state=42)

########################################
#     PREPARE DATA
########################################
features = ["market", "source_system", "account"]
label = "resolver_label"

feature_enc = OneHotEncoder(sparse=False)
feature_fit = feature_enc.fit(X=train[features])
X_tr = feature_enc.transform(X=train[features])
X_val = feature_enc.transform(X=val[features])

label_enc = OneHotEncoder(sparse=False)
label_fit = label_enc.fit(X=train[label].values.reshape(-1, 1))
y_tr = label_enc.transform(X=train[label].values.reshape(-1, 1))
y_val = label_enc.transform(X=val[label].values.reshape(-1, 1))

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
print(f"Accuracy with features = {features} = {accuracy :0.2%}")

########################################
#     PERSIST MODEL
########################################
model_dir = "models/"
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
print(f"After ser/deser, accuracy with features = {features} = {accuracy :0.2%}")

