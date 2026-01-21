import tensorflow as tf
from .tf_metrics import get_acc_tf, get_inference_time_tf, get_param_count_Deepsets_tf

def evaluate_deepsets_tf(model, train_loader, val_loader, test_loader, num_epochs=100, lr=0.0032):
    """Evaluates DeepSets models by training and computing performance metrics"""
    # Initialize training components
    criterion = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Train model
    validation_loss = train_tf(model, optimizer, criterion, 
                          train_loader, val_loader, 
                          num_epochs, patience=7)
    
    # Calculate metrics
    val_accuracy = get_acc_tf(model, val_loader)
    test_accuracy = get_acc_tf(model, test_loader)
    param_count = get_param_count_Deepsets_tf(model)
    inference_time = get_inference_time_tf(model, img_size=(1024, 8, 3))
    
    # Print results
    print(
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"Test Accuracy: {test_accuracy:.4f}, "
        f"Inference time: {inference_time:.4f}, "
        f"Validation Loss: {validation_loss:.4f}, "
        f"Parameter Count: {param_count}"
    )
    
    # Return metrics dictionary
    return {
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "val_loss": validation_loss,
        "inference_time": inference_time,
        "param_count": param_count
    }

def train_tf(model, optimizer, criterion, train_loader, valid_loader, num_epochs, patience=5):
    curr_patience = patience
    previous_epoch_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        for i, batch in enumerate(train_loader):
            with tf.GradientTape() as tape:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(targets, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Validation phase
        validation_loss = 0
        for batch in valid_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(targets, outputs)
            validation_loss += loss.numpy()

        validation_loss /= len(valid_loader)

        # Early Stopping Procedure
        if validation_loss < previous_epoch_loss:
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                break
        previous_epoch_loss = validation_loss

    return previous_epoch_loss
