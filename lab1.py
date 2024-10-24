import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=40)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)



model = Sequential()

model.add(Dense(10, activation='relu', input_dim=20))  

model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")