""" trains a model on X_train and generates y_test from X_test for submission.

Usage example :

python3 train_infer.py /pathto/preprocessed/X_train.csv /pathto/preprocessed/y_train.csv \\
    /pathto/preprocessed/X_test.csv --n_trees 4000 --max_depth 10 --learning_rate 0.1


"""

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import cudf
import pandas as pd
import numpy as np
import cupy as cp
import argparse

parser = argparse.ArgumentParser(description='Parses relevant arguments')
    
# Add arguments
parser.add_argument('path_X_train', type=str, help='path to preprocessed X_train csv')
parser.add_argument('path_y_train', type=str, help='path to preprocessed y_train csv')
parser.add_argument('path_X_test', type=str, help='path to preprocessed y_train csv')
parser.add_argument('--n_trees', type=int, default=4000, 
                    help='number of trees in the xgboost model')
parser.add_argument('--max_depth', type=int, default=12, 
                    help='Maximum depth of the trees in the model')
parser.add_argument('--learning_rate', type=float, default=0.13, 
                    help='learning_rate')

# Parse the arguments
args = parser.parse_args()


# Load data
X_train = pd.read_csv(args.path_X_train, sep=',', index_col='row_index')
y_train = pd.read_csv(args.path_y_train, sep=',', index_col='row_index')

# Split data
X_train_subset, X_val, y_train_subset, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=0)

# Convert data to GPU-compatible format
X_train_gpu = cudf.from_pandas(X_train_subset)
y_train_gpu = cudf.from_pandas(y_train_subset)
X_val_gpu = cudf.from_pandas(X_val)
y_val_gpu = cudf.from_pandas(y_val)

dtrain = xgb.DMatrix(X_train_gpu, label=y_train_gpu)
dval = xgb.DMatrix(X_val_gpu, label=y_val_gpu)



# Define initial parameters
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(np.unique(y_train)),  # Number of classes
    'tree_method': 'gpu_hist',  # Use GPU for training
    'eval_metric': 'mlogloss',  # Metric to minimize
    'learning_rate': args.learning_rate,
    'max_depth': args.max_depth,
    'max_bin': 256,
}


# Train the final model
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=args.n_trees,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=False
)

X_test = pd.read_csv(args.path_X_test, sep=',', index_col='row_index')

# Test evaluation
X_test_gpu = cudf.from_pandas(X_test)
dtest = xgb.DMatrix(X_test_gpu)
test_predictions = bst.predict(dtest)

# Convert predictions and true labels back to CPU for metrics
test_predictions = cp.asnumpy(test_predictions)

#utilitary

def decode_y(df):
    forward = {'Very Low': 0, 'Low': 1, 'Average': 2, 'High': 3, 'Very High': 4}
    backward = {v: k for k, v in forward.items()}
    df['piezo_groundwater_level_category'] = df['piezo_groundwater_level_category'].map(backward)
    return df

ix = X_test.index

df_pred = pd.DataFrame(test_predictions, index=ix, columns=['piezo_groundwater_level_category'])
df_pred
df_pred = decode_y(df_pred)
df_pred.to_csv('predictions.csv', index_label="row_index")

print(df_pred.head())