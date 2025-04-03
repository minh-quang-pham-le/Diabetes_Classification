# Step 1: Data Collection
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
import pickle

# Step 2: Statistics
data = pd.read_csv('diabetes.csv')

# profile = ProfileReport(data, title = "Diabetes Report", explorative=True)
# profile.to_file("statistics.html")

target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 3: Data preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 4: Model building

# Using LazyPredict to see which model is better with default parameters configuration
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)
models_lazypredict, predictions = clf.fit(x_train, x_test, y_train, y_test)

# Using GridCV to see which parameters configuration is best
models = {
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 0.5, 1, 2.5, 5, 10],
            "kernel": ["linear", "rbf", "poly", "sigmoid"]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {
            "C": [0.1, 0.5, 1, 2.5, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    }
}

# Step 5: Model Evaluation

# Evaluate LazyPredict
recalls = []
precisions = []
for name, model in clf.models.items():
    y_pred = model.predict(x_test)
    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))

models_lazypredict['Recall'] = recalls
models_lazypredict['Precision'] = precisions
print(models_lazypredict.sort_values(by="Recall", ascending=False))

# Evaluate GridSearchCV
print('\n')
results = []
for name, model in models.items():
    print(f"GridSearchCV cho {name}...")
    grid = GridSearchCV(model['model'], model['params'], cv=5, scoring='recall', verbose=0, n_jobs=4)
    grid.fit(x_train, y_train)

    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best score for {name}: {grid.best_score_}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    print("Score on test set:\n")
    print(classification_report(y_test, y_pred))

    # Store the results
    results.append({
        "Model": name,
        "Best params": grid.best_params_,
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("Results after GridSearchCV:\n")
print(results_df.sort_values(by = ['Recall'], ascending = False))

# Step 7: Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)