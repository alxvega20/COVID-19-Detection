import os
from preprocessing import load_data
from network import build_cnn
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report

# load data and preprocessing
BASE_DIR = "./data"
METADATA_CSV = os.path.join(BASE_DIR, "Chest_xray_Corona_Metadata.csv")
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train, y_train, X_test, y_test = load_data(BASE_DIR, METADATA_CSV)


# build and train model
model = build_cnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])


plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(OUTPUT_DIR, 'training_validation_accuracy.png'))
plt.close()

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(OUTPUT_DIR, 'training_validation_loss.png'))
plt.close()


# results
y_pred = (model.predict(X_test) > 0.5).astype("int32")


report = classification_report(y_test, y_pred, 
                                labels=[0, 1], 
                                target_names=["Normal", "Pnemonia"], 
                                output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))


test_loss, test_accuracy = model.evaluate(X_test, y_test)
with open(os.path.join(OUTPUT_DIR, 'final_test_metrics.txt'), 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")


# display random sample images with labels
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

correct_idxs = np.where(y_pred == y_test)[0]
incorrect_idxs = np.where(y_pred != y_test)[0]

print(f"Correct predictions: {len(correct_idxs)}")
print(f"Incorrect predictions: {len(incorrect_idxs)}")

num_samples = 5
correct_samples = random.sample(list(correct_idxs), min(num_samples, len(correct_idxs)))
incorrect_samples = random.sample(list(incorrect_idxs), min(num_samples, len(incorrect_idxs)))

all_samples = correct_samples + incorrect_samples
random.shuffle(all_samples) 

label_map = {0: "Normal", 1: "Pnemonia"}


plt.figure(figsize=(15, 10))

for i, idx in enumerate(all_samples):
    img = X_test[idx].reshape(224, 224)
    true_label = label_map[y_test[idx]]
    pred_label = label_map[y_pred[idx]]
    correct = (y_pred[idx] == y_test[idx])

    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"T: {true_label}\nP: {pred_label}\n{'✔️' if correct else '❌'}")
    plt.axis('off')

plt.suptitle('Sample Test Predictions', fontsize=20)
plt.tight_layout()
plt.savefig("./outputs/sample_predictions.png")
plt.show()