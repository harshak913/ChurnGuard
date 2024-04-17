import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def make_prediction(variable_labels,model_name):
  with open(model_name, 'rb') as file:
    loaded_model = pkl.load(file)

  keys_important = ['current_balance','current_month_debit','previous_month_debit','days_since_last_transaction','average_monthly_balance_prevQ','current_month_balance','previous_month_balance','average_monthly_balance_prevQ2','previous_month_end_balance','branch_code']
  values_important = [variable_labels[x] for x in keys_important]
  new_record = pd.DataFrame([values_important], columns=keys_important)

  if loaded_model.predict(new_record) == 1: 
    return "Customer has Churned"
  else:
    return "Customer has not Churned"
