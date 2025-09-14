
import time
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation, LeakyReLU, PReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal, GlorotUniform, HeUniform
from tensorflow.keras.regularizers import l2



# Funktion zum Setzen des Seeds in Python, NumPy und TensorFlow --> minimiert zufällige Schwankungen
def set_seed(seed=0):
    random.seed(seed); 
    np.random.seed(seed); 
    tf.keras.utils.set_random_seed(seed) 

# Hilfsklasse zum Messen der Epoche-Zeiten --> sodass wir "time_to_best" exakt bestimmen können
class EpochTimer(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch_durations_s = []                 # Trainingsbeginn festhalten
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()              # Epoche-Beginn festhalten
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_durations_s.append(time.perf_counter() - self._t0)  # Epoche-Ende festhalten

# Hilfsklasse zum Loggen der Lernrate (für optimizers mit Lernraten-Scheduling)
class LRLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []                               # Liste zum Speichern der Lernraten initialisieren

    def on_epoch_end(self, epoch, logs=None):       # Lernrate am Ende der Epoche speichern
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        lr_val = float(tf.keras.backend.get_value(lr))
        self.lrs.append(lr_val)


# Hilfsfunktion zur Anpassung der Eingabeformate --> MLP als flacher 2D Vektor und CNN als 3D Tensor mit Kanal
def _to_shape(x, is_cnn: bool):

    return x.reshape(-1, 28, 28, 1) if is_cnn else x.reshape(-1, 28 * 28)

# Datenvorbereitung
# Lädt MNIST, normalisiert, split in Train/Val, One-Hot-Encoding
def prepare_data(test_size: float = 0.1, seed: int = 0):

    # MNIST-Daten laden (60k Train, 10k Test)
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    # Normalisieren auf [0,1] und in float32 umwandeln
    x_train_raw = (x_train_raw / 255.0).astype("float32")
    x_test_raw  = (x_test_raw  / 255.0).astype("float32")

    # Train/Val-Split
    x_tr, x_val, y_tr_int, y_val_int = train_test_split(x_train_raw, y_train_raw, test_size=test_size, stratify=y_train_raw, random_state=seed)

    # One-Hot-Encoding
    y_tr   = to_categorical(y_tr_int,  num_classes=10)
    y_val  = to_categorical(y_val_int, num_classes=10)
    y_test = to_categorical(y_test_raw, num_classes=10)

    # Rückgabe der Trainings-, Validierungs- und Testdaten in entsprechender Form
    return x_tr, y_tr, x_val, y_val, x_test_raw, y_test

# Funktion zum Erstellen des MLP-Basismodells
def build_mlp_basic():

    model = Sequential()

    # Eingabe-Schicht
    model.add(Input(shape=(28 * 28,)))
    
    # Versteckte Schichten (manuell anpassbar)
    model.add(Dense(32, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(32, activation='relu', kernel_initializer=HeNormal()))
    
    # Ausgabe-Schicht
    model.add(Dense(10, activation='softmax', kernel_initializer=GlorotUniform()))
    
    return model

# Funktion zum Erstellen des CNN-Basismodells (Phase 1 & 2)
def build_cnn_basic():

    model = Sequential()

    # Eingabe-Schicht
    model.add(Input(shape=(28, 28, 1)))

    # CNN-Blöcke 
    model.add(Conv2D(16, (5, 5), activation='relu',
    padding='valid', kernel_initializer=HeNormal()))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
    padding='valid', kernel_initializer=HeNormal()))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(16, activation='relu', 
    kernel_initializer=HeNormal()))

    # Ausgabe-Schicht
    model.add(Dense(10, activation='softmax', 
    kernel_initializer=GlorotUniform()))

    return model

# Funktion zum Erstellen des erweiterten CNN-Models (Phase 3: Regularisierung)
def build_cnn_regularized():

    model = Sequential()

    # Eingabe-Schicht
    model.add(Input(shape=(28, 28, 1)))

    # CNN-Blöcke mit BatchNormalization und Dropout
    model.add(Conv2D(16, (5,5), padding='valid', kernel_initializer=HeNormal(), use_bias=False, activation=None)) # mit BatchNormalization use_bias = False
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), padding='valid', kernel_initializer=HeNormal(), use_bias=False, activation=None)) # mit BatchNormalization use_bias = False
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(16, kernel_initializer=HeNormal(), activation=None, use_bias=False))                            # mit BatchNormalization use_bias = False
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # Ausgabe-Schicht
    model.add(Dense(10, activation='softmax', kernel_initializer=GlorotUniform()))

    return model

# Funktion zum Kompilieren des Modells mit wählbarem Optimizer und Lernrate, CategoricalCrossentropy als Loss vorgegeben
def compile_model(model, optimizer="adam", lr=1e-3):

    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]) 

# Funktion zum Trainieren des Modells
# Generell wird das Training mit EarlyStopping (val_loss) und RestoreBestWeights durchgeführt
# Optional kann ReduceLROnPlateau für den Feinschliff aktiviert werden
# Es wird die Zeit für das gesamte Training und die Zeit bis zur Best-Epoch gemessen (über EpochTimer)
# Folgende Phasen haben unterschiedliche Einstellungen für das CNN-Model:
# Phase 1+2: epochs=20, patience=3, batch_size=64
# Phase 3: epochs=60, patience=5, batch_size=64, (128 bei Verwendung von BatchNormalization)
# Phase 4: batch_size=256
# MLP: epochs=20, patience=3, batch_size=64
# dass es vergleichbar bleibt, werden epochs=60 und patience=5 verwendet, wenn die optimierten Modelle trainiert werden 
def train(model, x_tr, y_tr, x_val, y_val, *, is_cnn: bool, batch_size=64, epochs=20, patience=3, verbose=0, seed=None, use_scheduler=False):

    # Seed setzen für Reproduzierbarkeit
    if seed is not None:
        set_seed(seed)

    # Richtige Form der Eingabedaten sicherstellen
    x_tr_  = _to_shape(x_tr,  is_cnn)
    x_val_ = _to_shape(x_val, is_cnn)

    # Callbacks: EarlyStopping + EpochTimer + LRLogger + (optional) ReduceLROnPlateau
    timer = EpochTimer()
    early = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    lrlog = LRLogger()
    callbacks = [early, timer, lrlog]
    if use_scheduler:
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)) # Lernrate halbieren, wenn sich val_loss 2 Epochen nicht verbessert

    t0 = time.perf_counter() # Zeitmessung Start
    history = model.fit(
        x_tr_, y_tr,
        validation_data=(x_val_, y_val),
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    train_time_s = time.perf_counter() - t0 # Gesamttrainingszeit inklusive Validierung

    # Kennzahlen zur Best-Epoch (min val_loss)
    h = history.history                             # History-Dictionary 
    best_idx = int(np.argmin(h["val_loss"]))        # Index der besten Epoche nach val_loss suchen
    best_epoch = best_idx + 1                       # Zählung ab 1, nicht ab 0
    
    # Train- und Val-Accuracy bei der Best-Epoch (kann None sein, wenn nicht im Modell verwendet)
    train_acc_at_best = h.get("accuracy", [None])[best_idx]
    val_acc_at_best   = h.get("val_accuracy", [None])[best_idx]

    # Zeit bis zur Best-Epoch (über Timer Callback) summieren
    time_to_best_s   = float(np.sum(timer.epoch_durations_s[:best_epoch]))

    # Generalization Gap (Train-Acc - Val-Acc) bei der Best-Epoch (Overfitting-Indikator)
    gen_gap_at_best = None
    if (train_acc_at_best is not None) and (val_acc_at_best is not None):
        gen_gap_at_best = float(train_acc_at_best - val_acc_at_best)

    return history, {
        "train_time_s": float(train_time_s),
        "best_epoch": best_epoch,
        "time_to_best_s": time_to_best_s,
        "generalization_gap": None if gen_gap_at_best is None else float(gen_gap_at_best),
        "epoch_durations_s": list(map(float, timer.epoch_durations_s)),
        "lr_per_epoch": list(map(float, lrlog.lrs))
    }

# Funktion zur Auswertung auf dem Validierungsset mit verschiedenen Metriken (Accuracy, F1, Parameteranzahl, Inferenzzeit pro Beispiel)
def evaluate_on_val(model, history, x_val, y_val, *, is_cnn: bool, infer_batch=128):

    # Richtige Form der Eingabedaten sicherstellen
    x_eval = _to_shape(x_val, is_cnn)

    # Loss und Accuracy auf den Validierungsdaten berechnen --> Loss wird nicht zurückgegeben
    _, acc = model.evaluate(x_eval, y_val, verbose=0)

    # F1-Score auf den Validierungsdaten berechnen
    y_true = np.argmax(y_val, axis=1)               # Wahre Klassenlabels (als Integer)
    y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)    # Vorhergesagte Klassenlabels (als Integer)
    f1m = f1_score(y_true, y_pred, average="macro")

    # Anzahl der Modell-Parameter ermitteln
    params = int(model.count_params())

    # Inferenzzeit messen (pro Beispiel in ms)
    _ = model.predict(x_eval[:min(len(x_eval), infer_batch)], batch_size=infer_batch, verbose=0) # Aufwärmen des Modells
    t0 = time.perf_counter()
    _ = model.predict(x_eval, batch_size=infer_batch, verbose=0) # Vallidierungsdaten vorhersagen
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) / x_eval.shape[0] * 1000.0    

    return {
        "accuracy": float(acc),
        "f1": float(f1m),
        "number_of_params": params,
        "inference_time_ms_for_example": float(infer_ms)
    }


