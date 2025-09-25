import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

print("Loading dataset...")
data_dict = pickle.load(open('./data.pickle','rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset loaded: {len(data)} samples, {len(labels)} labels")
print(f"Data shape: {data.shape}")
print(f"Unique labels: {np.unique(labels)}")

print("Splitting data into train and test sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Training set: {len(x_train)} samples")
print(f"Test set: {len(x_test)} samples")

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

print("Making predictions on test set...")
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('Accuracy: {:.2%} of samples classified correctly!'.format(score))

print("Saving model...")
f = open('model.p','wb')
pickle.dump({'model': model}, f)
f.close()
print("Model saved to model.p")

