# ============================================
# Heartbeat Classification using Simple RNN
# Author : Harshitha
# ============================================

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================================
# Load Dataset
# ============================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train = pd.read_csv("mitbih_train.csv", header=None)
test = pd.read_csv("mitbih_test.csv", header=None)

print("Training Shape :", train.shape)
print("Testing Shape  :", test.shape)

# ============================================
# Split Features and Labels
# ============================================

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

print("\nFeature Shape :", X_train.shape)

# ============================================
# Encode Labels
# ============================================

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

joblib.dump(encoder, "label_encoder.pkl")

print("\nClasses :", encoder.classes_)

# ============================================
# One Hot Encoding
# ============================================

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ============================================
# Reshape Data for RNN
# ============================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nInput Shape :", X_train.shape)

# ============================================
# Build Simple RNN Model
# ============================================

print("\nBuilding Model...")

model = Sequential()

model.add(

    SimpleRNN(

        units=64,

        activation="tanh",

        input_shape=(187,1)

    )

)

model.add(

    Dropout(0.30)

)

model.add(

    Dense(

        32,

        activation="relu"

    )

)

model.add(

    Dense(

        16,

        activation="relu"

    )

)

model.add(

    Dense(

        5,

        activation="softmax"

    )

)

# ============================================
# Compile Model
# ============================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Summary\n")

model.summary()

# ============================================
# Callbacks
# ============================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# ============================================
# Train Model
# ============================================

print("\nTraining Started...\n")

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test,y_test),

    epochs=20,

    batch_size=128,

    callbacks=[early_stop,checkpoint],

    verbose=1

)

# ============================================
# Evaluation
# ============================================

print("\nEvaluating Model...\n")

loss,accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print("=" * 60)
print("Test Accuracy : {:.2f}%".format(accuracy*100))
print("=" * 60)

# ============================================
# Predictions
# ============================================

predictions = model.predict(X_test)

predictions = np.argmax(predictions,axis=1)

actual = np.argmax(y_test,axis=1)

# ============================================
# Classification Report
# ============================================

print("\nClassification Report\n")

print(

    classification_report(

        actual,

        predictions

    )

)

# ============================================
# Confusion Matrix
# ============================================

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        actual,

        predictions

    )

)

# ============================================
# Save Final Model
# ============================================

model.save("rnn_model.keras")

print("\nModel Saved Successfully")

print("Saved Files")
print("-----------")
print("rnn_model.keras")
print("label_encoder.pkl")

print("\nTraining Completed Successfully")