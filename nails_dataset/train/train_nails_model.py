import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------
# Параметры
# ----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
DATASET_DIR = "nails_dataset/train"

# ----------------------
# Генератор данных
# ----------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Классы:", train_gen.class_indices)

# ----------------------
# Базовая модель MobileNetV2
# ----------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False  # заморожена для простоты

# ----------------------
# Полносвязная модель для бинарной классификации
# ----------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 0 = Healthy, 1 = Not Healthy
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------
# Обучение модели
# ----------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ----------------------
# Сохранение модели
# ----------------------
model.save("nails_model.h5")
print("Модель ногтей сохранена как nails_model.h5")
