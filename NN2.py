import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Shallow Network (1 hidden layer)
shallow = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
shallow.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Deep Network (3 hidden layers)
deep = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'), Dropout(0.3),
    Dense(256, activation='relu'), Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
deep.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train models
print("Training Shallow Model...")
h_shallow = shallow.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)
print("\nTraining Deep Model...")
h_deep = deep.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)

# Evaluate
_, acc_shallow = shallow.evaluate(x_test, y_test, verbose=0)
_, acc_deep = deep.evaluate(x_test, y_test, verbose=0)
print(f"\nShallow Model Accuracy: {acc_shallow:.4f}")
print(f"Deep Model Accuracy: {acc_deep:.4f}")

# Plot training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h_shallow.history['accuracy'], label='Shallow Train')
plt.plot(h_shallow.history['val_accuracy'], label='Shallow Val')
plt.plot(h_deep.history['accuracy'], label='Deep Train')
plt.plot(h_deep.history['val_accuracy'], label='Deep Val')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h_shallow.history['loss'], label='Shallow Train')
plt.plot(h_shallow.history['val_loss'], label='Shallow Val')
plt.plot(h_deep.history['loss'], label='Deep Train')
plt.plot(h_deep.history['val_loss'], label='Deep Val')
plt.title('Loss Comparison')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
plt.show()