import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    EPOCHS,
    LEARNING_RATE,
    CLASS_NAMES
)

from custom_dataset import load_dataframes, make_dataset
from cnn import build_cnn_model
from efficientnet import build_efficientnet_model


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(model_name):
    best_model_path = MODEL_DIR / f"{model_name}_best.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    return callbacks


def plot_training_graph(history, model_name):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(OUTPUT_DIR / f"{model_name}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(OUTPUT_DIR / f"{model_name}_loss.png")
    plt.close()


def evaluate_model(model, val_ds, model_name):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        predicted_labels = predictions.argmax(axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    print(f"\n{model_name} Classification Report")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print(f"\n{model_name} Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    return metrics


def train_single_model(model_name, model, train_ds, val_ds):
    print(f"\nTraining started: {model_name}")

    model = compile_model(model)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(model_name)
    )

    plot_training_graph(history, model_name)

    metrics = evaluate_model(model, val_ds, model_name)

    final_path = MODEL_DIR / f"{model_name}_final.keras"
    model.save(final_path)

    print(f"{model_name} final model saved at: {final_path}")

    return metrics


def train():
    print("Loading dataset...")

    train_df, val_df = load_dataframes(
        use_subset=True,
        samples_per_class=5000
    )

    print("Train images:", len(train_df))
    print("Validation images:", len(val_df))

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    results = []

    cnn_model = build_cnn_model()
    cnn_metrics = train_single_model(
        "cnn",
        cnn_model,
        train_ds,
        val_ds
    )
    results.append(cnn_metrics)

    efficientnet_model = build_efficientnet_model()
    efficientnet_metrics = train_single_model(
        "efficientnet",
        efficientnet_model,
        train_ds,
        val_ds
    )
    results.append(efficientnet_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nFinal Model Comparison")
    print(results_df)


if __name__ == "__main__":
    train()