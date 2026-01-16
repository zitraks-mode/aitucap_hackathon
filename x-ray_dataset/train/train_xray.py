import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------
# Параметры
# ----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATASET_DIR = "x-ray_dataset/train"

# ----------------------
# Генератор данных
# ----------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # ВАЖНО
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes
print("Классы:", train_gen.class_indices)

# ----------------------
# Базовая модель
# ----------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ----------------------
# Полная модель
# ----------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------
# Обучение
# ----------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ----------------------
# Сохранение
# ----------------------
model.save("xray_model.h5")
print("X-ray модель сохранена как xray_model.h5")
