from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Image preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_set = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_set.num_classes
class_names = list(train_set.class_indices.keys())

# Extract plant leaf names
leaf_names = [name.split("_")[0] for name in class_names]
leaf_unique = sorted(list(set(leaf_names)))
leaf_to_index = {leaf:i for i,leaf in enumerate(leaf_unique)}

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_set, validation_data=test_set, epochs=10)

os.makedirs("model", exist_ok=True)
model.save("model/plant_disease_model.h5")

print("✅ Model saved successfully!")

# ----------------------------
# PLOT TRAINING ACCURACY
# ----------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Validation"])
plt.title("Training vs Validation Accuracy")
plt.show()

# ----------------------------
# LEAF TYPE ACCURACY
# ----------------------------
predictions = model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes

true_leaf = np.array([leaf_to_index[name.split("_")[0]] for name in class_names])[true_classes]
pred_leaf = np.array([leaf_to_index[class_names[p].split("_")[0]] for p in predicted_classes])

leaf_accuracy = np.mean(true_leaf == pred_leaf) * 100
print(f"\n🍃 Leaf Type Detection Accuracy: {leaf_accuracy:.2f}%")

# ----------------------------
# DISEASE ACCURACY
# ----------------------------
disease_accuracy = np.mean(true_classes == predicted_classes) * 100
print(f"🧠 Disease Detection Accuracy: {disease_accuracy:.2f}%")

# ----------------------------
# CLASS-WISE DISEASE ACCURACY
# ----------------------------
for i, name in enumerate(class_names):
    mask = true_classes == i
    acc = np.mean(predicted_classes[mask] == i) * 100
    print(f"Class: {name} → {acc:.2f}%")
