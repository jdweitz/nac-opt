import time
import tensorflow as tf

def get_acc_tf(model, dataloader):
    correct = 0
    total = 0
    for data, targets in dataloader:
        outputs = model(data)
        predicted = tf.argmax(outputs, axis=1)
        true_labels = tf.argmax(targets, axis=1)
        correct += tf.reduce_sum(tf.cast(predicted == true_labels, tf.float32))
        total += len(targets)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy.numpy()

def get_inference_time_tf(model, img_size=(1024, 8, 3)):
    x = tf.random.normal(img_size)
    start = time.time()
    for _ in range(100):
        y = model(x)
    end = time.time()
    return end - start

def get_param_count_Deepsets_tf(model):
    return model.count_params()
