import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

#LOAD DATASET INTO MEMORY)
(train_ds, test_ds), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


#PREPROCESS AND SPLIT DATASET INTO POOL, TRAIN AND TEST SETS
train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
train_ds = train_ds.batch(64)
train_ds, pool = tf.keras.utils.split_dataset(train_ds, left_size=0.2, shuffle=True, seed=42)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(64)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


#TRAIN MODEL ON TRAINING SET
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
hist = model.fit(train_ds, epochs=25, validation_data=test_ds)

#SAVE METRICS 
plt.plot(hist.history['loss'], label="Training loss")
plt.plot(hist.history['val_loss'], label="Validation loss")
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(hist.history['sparse_categorical_accuracy'], label="Accuracy")
plt.plot(hist.history['val_sparse_categorical_accuracy'], label="Validation accuracy")
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#EVALUTE ON POOL 

#PICK FROM POOL 

#GO AGANE 

#COMPARE METRICS AND SHOW GRAPHS
