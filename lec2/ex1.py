from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning


from sklearn.model_selection import RandomizedSearchCV



data = load_breast_cancer()



x = data.data
y = data.target



"""
NORMAL MODEL

model = LogisticRegression(max_iter=1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print( accuracy)


test_pred = model.predict(x_test)
print( test_pred)
"""


"""


GRIDSEARCH


param_grid={'C': [0.1, 1, 10]}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000, penalty='l2'), param_grid=param_grid, cv=5 )


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    grid_search.fit(x, y)


print(grid_search.cv_results_)
print(grid_search.best_params_)


"""

param_dist = {
    "l1_ratio": (0, 1),
    "C": [0.1, 1, 10],
    "max_iter": [100, 500, 1000],
    "penalty": ['l1', 'l2', 'elasticnet', 'none']
}


n_iter_search = 30
random_search = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring="roc_auc_ovr",
    random_state=42,
)

random_search.fit(x, y)

print(random_search.best_params_)




model = LogisticRegression(**random_search.best_params_)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print( accuracy)

