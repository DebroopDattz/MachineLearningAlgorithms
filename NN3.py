import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
y_train,y_test=to_categorical(y_train,10),to_categorical(y_test,10)

model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(512,activation='relu'),BatchNormalization(),Dropout(0.3),
    Dense(256,activation='relu'),BatchNormalization(),Dropout(0.3),
    Dense(128,activation='relu'),BatchNormalization(),Dropout(0.3),
    Dense(10,activation='softmax')
])

model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=15,batch_size=128,validation_split=0.2,verbose=1)

loss,acc=model.evaluate(x_test,y_test,verbose=0)
print(f"Accuracy={acc}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Validation')

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train')
plt.plot(history.history['val_loss'],label='Validation')
plt.show()