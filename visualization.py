import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    RocCurveDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

saved_results_dir = "/app/HAM10000/results"

models = {"inceptionv3": 1, "vgg16": 2, "resnet50": 3}

plots_dir = os.path.join(saved_results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

metrics_dict = {
    metric: {"a": [], "b": [], "c": []}
    for metric in ["accuracy", "precision", "recall", "f1_score", "specificity", "auc"]
}

for model in models.keys():

    # Training history plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, abc in enumerate(["a", "b", "c"]):
        exp = f"{model}_exp{models[model]}{abc}"
        exp_name = f"Exp{models[model]}{abc}"
        history_path = os.path.join(saved_results_dir, f"{exp}_history.pkl")

        with open(history_path, "rb") as f:
            history = pickle.load(f)

        # Training and validation loss
        axes[0, idx].plot(history["loss"], label="Train Loss")
        axes[0, idx].plot(history["val_loss"], label="Val Loss", linestyle="--")
        axes[0, idx].set_title(f"{exp_name} Loss")
        axes[0, idx].set_xlabel("Epochs")
        axes[0, idx].set_ylabel("Loss")
        axes[0, idx].legend()
        axes[0, idx].grid(True)

        # Training and validation accuracy
        axes[1, idx].plot(history["accuracy"], label="Train Accuracy")
        axes[1, idx].plot(history["val_accuracy"], label="Val Accuracy", linestyle="--")
        axes[1, idx].set_title(f"{exp_name} Accuracy")
        axes[1, idx].set_xlabel("Epochs")
        axes[1, idx].set_ylabel("Accuracy")
        axes[1, idx].legend()
        axes[1, idx].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{model}_training_loss.png"))
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    for idx, abc in enumerate(["a", "b", "c"]):
        exp = f"{model}_exp{models[model]}{abc}"
        exp_name = f"Exp{models[model]}{abc}"

        true_classes_path = os.path.join(
            saved_results_dir, f"{exp}_eval_true_classes.pkl"
        )
        predictions_path = os.path.join(
            saved_results_dir, f"{exp}_eval_predictions.pkl"
        )

        with open(true_classes_path, "rb") as f:
            true_classes = pickle.load(f)

        with open(predictions_path, "rb") as f:
            predictions = pickle.load(f)

        predicted_classes = (predictions >= 0.5).astype(int)
        cm = confusion_matrix(true_classes, predicted_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False)
        axes[idx].set_title(f"{exp_name} Confusion Matrix")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{model}_confusion_matrices.png"))
    plt.close(fig)

    # ROC curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, abc in enumerate(["a", "b", "c"]):
        exp = f"{model}_exp{models[model]}{abc}"
        exp_name = f"Exp{models[model]}{abc}"

        metrics_path = os.path.join(saved_results_dir, f"{exp}_metrics.pkl")
        predictions_path = os.path.join(
            saved_results_dir, f"{exp}_eval_predictions.pkl"
        )
        true_classes_path = os.path.join(
            saved_results_dir, f"{exp}_eval_true_classes.pkl"
        )

        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
        with open(predictions_path, "rb") as f:
            predictions = pickle.load(f)
        with open(true_classes_path, "rb") as f:
            true_classes = pickle.load(f)

        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)

        # Collect metrics by experiment type
        for metric in metrics_dict.keys():
            metrics_dict[metric][abc].append(metrics[metric])

        RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            estimator_name=exp,
        ).plot(ax=axes[idx])
        axes[idx].set_title(f"{exp_name} ROC Curve")
        axes[idx].legend()
        axes[idx].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{model}_roc_curves.png"))
    plt.close(fig)


model_sums = {"inceptionv3": [], "vgg16": [], "resnet50": []}
experiment_sums = {"a": [], "b": [], "c": []}
metric_count = len(metrics_dict)

# List of metrics to plot in each image
metrics_list = list(metrics_dict.keys())

# Iterate over each set of 3 metrics
for i in range(0, len(metrics_list), 3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for j, metric in enumerate(metrics_list[i : i + 3]):
        labels = ["Exp a", "Exp b", "Exp c"]
        data_a = [metrics_dict[metric]["a"][k] for k in range(len(labels))]
        data_b = [metrics_dict[metric]["b"][k] for k in range(len(labels))]
        data_c = [metrics_dict[metric]["c"][k] for k in range(len(labels))]

        # Append data for average calculations
        model_sums["inceptionv3"].extend(data_a)
        model_sums["vgg16"].extend(data_b)
        model_sums["resnet50"].extend(data_c)
        experiment_sums["a"].append(np.mean(data_a))
        experiment_sums["b"].append(np.mean(data_b))
        experiment_sums["c"].append(np.mean(data_c))

        x = np.arange(len(labels))
        width = 0.15

        ax = axes[j]

        rects1 = ax.bar(x - width, data_a, width, label="inceptionv3")
        rects2 = ax.bar(x, data_b, width, label="vgg16")
        rects3 = ax.bar(x + width, data_c, width, label="resnet50")

        ax.set_xlabel("Experiment Type")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"metrics_{i//3 + 1}_barplot.png"))
    plt.close(fig)

# Calculate averages
average_model_metrics = {model: np.mean(model_sums[model]) for model in model_sums}
average_experiment_metrics = {
    exp: np.mean(experiment_sums[exp]) for exp in experiment_sums
}

# Print average metrics
print("Average metrics per model:")
for model, avg in average_model_metrics.items():
    print(f"{model}: {avg:.4f}")

print("\nAverage metrics per experiment:")
for exp, avg in average_experiment_metrics.items():
    print(f"Experiment {exp}: {avg:.4f}")
