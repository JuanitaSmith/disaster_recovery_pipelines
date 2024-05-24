from mlsmote import MLSMOTE

# Load your data here
X_train, y_train = data

# indicate categorical features column index as it can be more than one, so infer it from your data
categorical_features = [0]

# Create an instance of MLSMOTE
mlsmote = MLSMOTE(categorical_features=categorical_features,input_columns=X_train.columns, label_columns=y_train.columns,random_state=77)