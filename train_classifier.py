import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print(f"\nAccuracy: {score * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predict))

print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Save model + label info
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'labels': model.classes_
    }, f)

print("\nModel saved as model.p")
