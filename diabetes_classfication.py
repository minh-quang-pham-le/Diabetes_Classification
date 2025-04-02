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
from sklearn.metrics import classification_report

# Step 2: Statistics
data = pd.read_csv('diabetes.csv')

# profile = ProfileReport(data, title = "Diabetes Report", explorative=True)
# profile.to_file("diabetes.html")

target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 3: Data preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 4: Model building
models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier()
}

# Step 5: Model Evaluation
for name, model in models.items():
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_predict))
