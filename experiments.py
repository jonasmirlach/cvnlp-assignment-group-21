import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_dir = "/app/HAM10000"

train_data_dir = os.path.join(base_dir, "train")
test_data_dir = os.path.join(base_dir, "test")

train_metadata = os.path.join(train_data_dir, "metadata.csv")
test_metadata = os.path.join(test_data_dir, "metadata.csv")

saved_results_dir = os.path.join(base_dir, "results")


# Function to get image paths and labels
def get_image_paths_and_labels(
    metadata_path: str, data_dir: str
) -> Tuple[List[str], np.ndarray]:
    metadata_df = pd.read_csv(metadata_path)
    image_paths = [
        os.path.join(data_dir, "images", f"{image_id}.jpg")
        for image_id in metadata_df["image_id"]
    ]
    labels = metadata_df["dx"].values
    return image_paths, labels


# Custom HAM1000 data generator
class HAM1000DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        batch_size: int,
        image_size: Tuple[int, int],
        is_training: bool = True,
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.indices = np.arange(len(self.image_paths))
        if is_training:
            np.random.shuffle(self.indices)

        # Map all labels from 7 classes to the 2 classes
        label_mapping = {
            "akiec": 1,
            "mel": 1,
            "bcc": 1,
            "bkl": 0,
            "df": 0,
            "nv": 0,
            "vasc": 0,
        }
        self.numeric_labels = [label_mapping[label] for label in labels]

        # Data augmentations
        if self.is_training:
            self.datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=15,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="nearest",
            )
        else:
            self.datagen = ImageDataGenerator(rescale=1.0 / 255)

    def __len__(self) -> int:
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:

        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.numeric_labels[i] for i in batch_indices]

        batch_images = [self.load_image(image_path) for image_path in batch_image_paths]
        batch_images = np.array(batch_images)

        # Apply augmentations
        batch_images, batch_labels = next(
            self.datagen.flow(batch_images, batch_labels, batch_size=self.batch_size)
        )

        return batch_images, np.array(batch_labels)

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.indices)

    def load_image(self, image_path: str) -> np.ndarray:

        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = image.numpy()
        image = image.astype(np.uint8)

        # Mean Filtering (Noise Reduction)
        image = cv2.blur(image, (2, 2))

        # Histogram Equalization
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        return image


# Get image paths and labels
train_image_paths, train_labels = get_image_paths_and_labels(
    train_metadata, train_data_dir
)
test_image_paths, test_labels = get_image_paths_and_labels(test_metadata, test_data_dir)


def train(
    model: Model,
    model_save_name: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    epochs: int = 30,
    learning_rate: float = 1e-4,
) -> tf.keras.callbacks.History:

    np.random.seed(42)
    tf.random.set_seed(42)

    # Creat train val split in the ration 80:20
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_image_paths,
        train_labels,
        train_size=0.8,
        stratify=train_labels,
        random_state=42,
    )

    train_generator = HAM1000DataGenerator(
        tr_paths, tr_labels, batch_size, image_size, is_training=True
    )
    val_generator = HAM1000DataGenerator(
        val_paths, val_labels, batch_size, image_size, is_training=False
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy", tf.keras.metrics.Precision()],
    )

    # Ensure equal class distribution in training
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_generator.numeric_labels),
        y=train_generator.numeric_labels,
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Include early stopping in the training process
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    callbacks = [early_stopping]

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=0.000001
    )
    callbacks.append(reduce_lr)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        # class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # Save model and history
    model.save(os.path.join(saved_results_dir, f"{model_save_name}.h5"))
    with open(
        os.path.join(saved_results_dir, f"{model_save_name}_history.pkl"), "wb"
    ) as f:
        pickle.dump(history.history, f)

    return history


def evaluate_training(history: dict) -> None:

    print(history)

    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


def evaluate(
    model: Model,
    model_save_name: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, float]:

    test_generator = HAM1000DataGenerator(
        test_image_paths, test_labels, batch_size, image_size, is_training=False
    )

    predictions = []
    true_classes = []

    for step in tqdm(range(len(test_generator)), desc="Evaluating", unit="batch"):
        x, y = test_generator[step]
        preds = model.predict_on_batch(x)
        true_classes.extend(y)
        predictions.append(preds)

    true_classes = np.array(true_classes)
    predictions = np.concatenate(predictions).flatten()
    predicted_classes = (predictions >= 0.5).astype(int)

    precision = precision_score(true_classes, predicted_classes)
    accuracy = accuracy_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)
    auc = roc_auc_score(true_classes, predictions)
    cm = confusion_matrix(true_classes, predicted_classes)

    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "specificity": specificity,
    }

    print(metrics)

    with open(
        os.path.join(saved_results_dir, f"{model_save_name}_metrics.pkl"), "wb"
    ) as f:
        pickle.dump(metrics, f)
    with open(
        os.path.join(saved_results_dir, f"{model_save_name}_eval_true_classes.pkl"),
        "wb",
    ) as f:
        pickle.dump(true_classes, f)
    with open(
        os.path.join(saved_results_dir, f"{model_save_name}_eval_predictions.pkl"), "wb"
    ) as f:
        pickle.dump(predictions, f)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return metrics


def add_binary_classification_head(model: Model) -> Model:
    model.add(Flatten(name="flattened"))
    model.add(Dropout(0.2, name="dropout1"))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2, name="dropout2"))
    model.add(Dense(1, activation="sigmoid", name="predictions"))
    return model


def experiment(
    model: str,
    pre_trained_weights: bool,
    backbone_trainable: bool,
    exp_name: str,
    learning_rate: float,
) -> Model:

    weights = "imagenet" if pre_trained_weights else None

    if model == "inceptionv3":
        backbone = tf.keras.applications.InceptionV3(
            weights=weights, include_top=False, input_shape=(224, 224, 3)
        )
    elif model == "vgg16":
        backbone = tf.keras.applications.VGG16(
            weights=weights, include_top=False, input_shape=(224, 224, 3)
        )
    elif model == "resnet50":
        backbone = tf.keras.applications.ResNet50V2(
            weights=weights, include_top=False, input_shape=(224, 224, 3)
        )

    full_model = Sequential()
    full_model.add(backbone)
    full_model.build(input_shape=(None, 224, 224, 3))
    full_model = add_binary_classification_head(full_model)

    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = backbone_trainable

    full_model.summary()

    history = train(
        full_model,
        model_save_name=f"{model}_{exp_name}",
        epochs=30,
        learning_rate=learning_rate,
    )
    evaluate_training(history.history)
    evaluate(full_model, model_save_name=f"{model}_{exp_name}")


def exp1a():
    experiment(
        model="inceptionv3",
        pre_trained_weights=False,
        backbone_trainable=True,
        exp_name="exp1a",
        learning_rate=0.01,
    )


def exp1b():
    experiment(
        model="inceptionv3",
        pre_trained_weights=True,
        backbone_trainable=False,
        exp_name="exp1b",
        learning_rate=0.0001,
    )


def exp1c():
    experiment(
        model="inceptionv3",
        pre_trained_weights=True,
        backbone_trainable=True,
        exp_name="exp1c",
        learning_rate=0.0001,
    )


def exp2a():
    experiment(
        model="vgg16",
        pre_trained_weights=False,
        backbone_trainable=True,
        exp_name="exp2a",
        learning_rate=0.01,
    )


def exp2b():
    experiment(
        model="vgg16",
        pre_trained_weights=True,
        backbone_trainable=False,
        exp_name="exp2b",
        learning_rate=0.0001,
    )


def exp2c():
    experiment(
        model="vgg16",
        pre_trained_weights=True,
        backbone_trainable=True,
        exp_name="exp2c",
        learning_rate=0.0001,
    )


def exp3a():
    experiment(
        model="resnet50",
        pre_trained_weights=False,
        backbone_trainable=True,
        exp_name="exp3a",
        learning_rate=0.01,
    )


def exp3b():
    experiment(
        model="resnet50",
        pre_trained_weights=True,
        backbone_trainable=False,
        exp_name="exp3b",
        learning_rate=0.0001,
    )


def exp3c():
    experiment(
        model="resnet50",
        pre_trained_weights=True,
        backbone_trainable=True,
        exp_name="exp3c",
        learning_rate=0.0001,
    )


if __name__ == "__main__":
    exp1a()
    exp1b()
    exp1c()
    exp2a()
    exp2b()
    exp2c()
    exp3a()
    exp3b()
    exp3c()
