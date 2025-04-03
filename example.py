import pickle
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

test_data = [[1,85,66,29,0,26.6,0.351,31]]
y_pred = model.predict(test_data)
print(y_pred)