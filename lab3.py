import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=40)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()


model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.3)) 

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(1, activation='sigmoid'))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


history = model.fit(
    X_train, y_train, 
    epochs=100,  
    batch_size=16,  
    validation_data=(X_test, y_test)
)


test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\nTest Lab3 Accuracy: {test_acc:.4f}")
