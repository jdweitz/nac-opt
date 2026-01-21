import numpy as np
import tensorflow as tf

def setup_data_loaders_deepsets_tf(base_file_name, batch_size=32):
    train_data_files = [f"./data/normalized_data3/x_train_{base_file_name}.npy"]
    train_target_files = [f"./data/normalized_data3/y_train_{base_file_name}.npy"]
    test_data_files = [f"./data/normalized_data3/x_test_{base_file_name}.npy"]
    test_target_files = [f"./data/normalized_data3/y_test_{base_file_name}.npy"]

    x_train = np.load(train_data_files[0])
    y_train = np.load(train_target_files[0])
    x_test = np.load(test_data_files[0])
    y_test = np.load(test_target_files[0])

    # train val split
    train_end = int(0.8 * len(x_train))
    x_val = x_train[train_end:]
    y_val = y_train[train_end:]
    x_train = x_train[:train_end]
    y_train = y_train[:train_end]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_loader = train_dataset.batch(batch_size)
    val_loader = val_dataset.batch(batch_size)
    test_loader = test_dataset.batch(batch_size)

    return train_loader, val_loader, test_loader
