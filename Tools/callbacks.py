import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, num_classes):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        val_true = []
        val_pred = []

        for x_batch, y_batch in self.validation_data:
            y_pred_batch = self.model.predict(x_batch, verbose=0)
            y_pred_class = np.argmax(y_pred_batch, axis=1)
            y_true_class = np.argmax(y_batch, axis=1)

            val_pred.extend(y_pred_class)
            val_true.extend(y_true_class)

        f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
        p = precision_score(val_true, val_pred, average='macro', zero_division=0)
        r = recall_score(val_true, val_pred, average='macro', zero_division=0)

        print(f'\nEpoch {epoch+1} - val_macro_f1: {f1:.4f} | val_macro_precision: {p:.4f} | val_macro_recall: {r:.4f}')

        if logs is not None:
            logs['val_macro_f1'] = f1
            logs['val_macro_precision'] = p
            logs['val_macro_recall'] = r


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.best_f1 = -np.Inf
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        current_f1 = logs.get("val_macro_f1")
        if current_f1 is not None and current_f1 > self.best_f1:
            print(f"\n[SaveBestModel] val_macro_f1 improved from {self.best_f1:.4f} to {current_f1:.4f}. Saving model.")
            self.best_f1 = current_f1
            self.model.save(self.save_path)


class EarlyStoppingOnF1(tf.keras.callbacks.Callback):
    def __init__(self, patience=5):
        super().__init__()
        self.patience = patience
        self.best_f1 = -np.Inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_f1 = logs.get("val_macro_f1")
        if current_f1 is None:
            return

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
        else:
            self.wait += 1
            print(f"[EarlyStoppingOnF1] No improvement in val_macro_f1 for {self.wait} epochs.")
            if self.wait >= self.patience:
                print("[EarlyStoppingOnF1] Stopping training.")
                self.model.stop_training = True


class ReduceLROnF1(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.2, patience=2, min_lr=1e-7):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_f1 = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_f1 = logs.get("val_macro_f1")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        if current_f1 is None:
            return

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"[ReduceLROnF1] Reducing learning rate to {new_lr:.1e}")
                self.wait = 0

