from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn import datasets





digits = datasets.load_digits()
x = digits.data
y = digits.target


"""
model = LogisticRegression(max_iter=10000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))



"""




param_dist = {
    "l1_ratio": (0, 1),
    "C": [0.1, 1, 2],
    "max_iter": [100, 500, 1000, 5000],
    "penalty": ['l1', 'l2', 'elasticnet', 'none']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_dist)


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    grid_search.fit(x, y)


print(grid_search.cv_results_)
print(grid_search.best_params_)




model = LogisticRegression(**grid_search.best_params_)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))

print(y_pred)
print(y_test)