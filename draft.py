import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

tf.random.set_seed(42)
np.random.seed(42)

#LOAD DATASET INTO MEMORY
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

train_ds, pool_ds = tf.keras.utils.split_dataset(train_ds, left_size=0.05, shuffle=True, seed=42)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(64)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

initial_train_data = list(train_ds.unbatch().as_numpy_iterator())
initial_pool_data = list(pool_ds.unbatch().as_numpy_iterator())

def to_dataset(data, batch_size=64, shuffle=True):
    images, labels = zip(*data)
    ds = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    if shuffle:
        ds = ds.shuffle(len(data))
    return ds.batch(batch_size)

#QUERY STRATEGY FUNCTIONS
def query_least_confidence(preds, n=64):
    uncertainties = 1 - np.max(preds, axis=1)
    query_indices = np.argsort(uncertainties)[-n:]
    return query_indices

def query_margin_sampling(preds, n=64):
    sorted_probs = np.sort(preds, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    query_indices = np.argsort(margins)[:n]
    return query_indices

def query_entropy_sampling(preds, n=64):
    entropy = -np.sum(preds * np.log(preds + 1e-10), axis=1)
    query_indices = np.argsort(entropy)[-n:]
    return query_indices

def query_random_sampling(preds, n=64):
    indices = np.arange(0, len(pool_data_random))
    query_indices = np.random.choice(indices, size=n, replace=False)
    return query_indices

#CREATE MODEL FUNCTION
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

#INIT MODELS AND DATA FOR EACH STRATEGY
model_lc      = create_model()
model_ms      = create_model()
model_entropy = create_model()
model_random  = create_model()

train_data_lc      = initial_train_data.copy()
pool_data_lc       = initial_pool_data.copy()

train_data_ms      = initial_train_data.copy()
pool_data_ms       = initial_pool_data.copy()

train_data_entropy = initial_train_data.copy()
pool_data_entropy  = initial_pool_data.copy()

train_data_random = initial_train_data.copy()
pool_data_random  = initial_pool_data.copy()

#LEARNING LOOP AND PARAMS
n_rounds = 50
query_size = 32
epochs_per_round = 15

#LISTS TO STORE ACCURACY
performance_lc = []
performance_ms = []
performance_entropy = []
performance_random = []

for round in range(n_rounds):
    print(f"=== Round {round+1}/{n_rounds} ===")
    
    #LEAST CONFIDENC
    if len(pool_data_lc) >= query_size:
        pool_images, pool_labels = zip(*pool_data_lc)
        pool_images = np.array(pool_images)
        preds = model_lc.predict(pool_images, verbose=0)
        query_idx = query_least_confidence(preds, n=query_size)
        selected = [pool_data_lc[i] for i in query_idx]
        train_data_lc.extend(selected)
        pool_data_lc = [pool_data_lc[i] for i in range(len(pool_data_lc)) if i not in query_idx]
        updated_train_ds = to_dataset(train_data_lc)
        model_lc.fit(updated_train_ds, epochs=epochs_per_round, verbose=0)
        loss, acc = model_lc.evaluate(test_ds, verbose=0)
        performance_lc.append(acc)
        print(f"Least Confidence Test Accuracy: {acc:.4f}")
    
    #MARGIN SAMPLING
    if len(pool_data_ms) >= query_size:
        pool_images, pool_labels = zip(*pool_data_ms)
        pool_images = np.array(pool_images)
        preds = model_ms.predict(pool_images, verbose=0)
        query_idx = query_margin_sampling(preds, n=query_size)
        selected = [pool_data_ms[i] for i in query_idx]
        train_data_ms.extend(selected)
        pool_data_ms = [pool_data_ms[i] for i in range(len(pool_data_ms)) if i not in query_idx]
        updated_train_ds = to_dataset(train_data_ms)
        model_ms.fit(updated_train_ds, epochs=epochs_per_round, verbose=0)
        loss, acc = model_ms.evaluate(test_ds, verbose=0)
        performance_ms.append(acc)
        print(f"Margin Sampling Test Accuracy: {acc:.4f}")
    
    #ENTROPY SAMPLING
    if len(pool_data_entropy) >= query_size:
        pool_images, pool_labels = zip(*pool_data_entropy)
        pool_images = np.array(pool_images)
        preds = model_entropy.predict(pool_images, verbose=0)
        query_idx = query_entropy_sampling(preds, n=query_size)
        selected = [pool_data_entropy[i] for i in query_idx]
        train_data_entropy.extend(selected)
        pool_data_entropy = [pool_data_entropy[i] for i in range(len(pool_data_entropy)) if i not in query_idx]
        updated_train_ds = to_dataset(train_data_entropy)
        model_entropy.fit(updated_train_ds, epochs=epochs_per_round, verbose=0)
        loss, acc = model_entropy.evaluate(test_ds, verbose=0)
        performance_entropy.append(acc)
        print(f"Entropy Sampling Test Accuracy: {acc:.4f}")

    #RANDOM SAMPLING
    if len(pool_data_random) >= query_size:
        pool_images, pool_labels = zip(*pool_data_random)
        pool_images = np.array(pool_images)
        preds = model_random.predict(pool_images, verbose=0)
        query_idx = query_random_sampling(preds, n=query_size)
        selected = [pool_data_random[i] for i in query_idx]
        train_data_random.extend(selected)
        pool_data_random = [pool_data_random[i] for i in range(len(pool_data_random)) if i not in query_idx]
        updated_train_ds = to_dataset(train_data_random)
        model_random.fit(updated_train_ds, epochs=epochs_per_round, verbose=0)
        loss, acc = model_random.evaluate(test_ds, verbose=0)
        performance_random.append(acc)
        print(f"Random Sampling Test Accuracy: {acc:.4f}")

#VISUALIZE TO COMPARE
lc_last10      = performance_lc[-10:]
ms_last10      = performance_ms[-10:]
entropy_last10 = performance_entropy[-10:]
random_last10  = performance_random[-10:]

def compare_strategies(data1, data2, name1, name2):
    #T-TEST
    t_stat, p_val = stats.ttest_rel(data1, data2)
    #WILCOXON TEST
    w_stat, p_val_w = stats.wilcoxon(data1, data2)
    print(f"Comparison: {name1} vs {name2}")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    print(f"  Wilcoxon test: stat = {w_stat:.3f}, p = {p_val_w:.3f}")
    print()

compare_strategies(lc_last10, ms_last10, "Least Confidence", "Margin Sampling")
compare_strategies(lc_last10, entropy_last10, "Least Confidence", "Entropy Sampling")
compare_strategies(lc_last10, random_last10, "Least Confidence", "Random Sampling")
compare_strategies(ms_last10, entropy_last10, "Margin Sampling", "Entropy Sampling")
compare_strategies(ms_last10, random_last10, "Margin Sampling", "Random Sampling")
compare_strategies(entropy_last10, random_last10, "Entropy Sampling", "Random Sampling")

rounds = np.arange(1, n_rounds + 1)
plt.figure(figsize=(8,6))
plt.plot(rounds, performance_lc, marker='o', label="Least Confidence")
plt.plot(rounds, performance_ms, marker='o', label="Margin Sampling")
plt.plot(rounds, performance_entropy, marker='o', label="Entropy Sampling")
plt.plot(rounds, performance_random, marker='o', label="Random Sampling")
plt.xlabel("Active Learning Round")
plt.ylabel("Test Accuracy")
plt.title("Performance Comparison of Query Strategies")
plt.legend()
plt.show()
