import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

from model_train import _to_shape




# Funktion zur Darstellung der Lernkurven
# Zeigt Loss und Accuracy jeweils für Train und Val
def plot_learning_curves(history, title="Learning Curves"):

    # Diktionary aus History extrahieren 
    h = history.history
    loss = h.get("loss", [])
    val_loss = h.get("val_loss", [])
    acc = h.get("accuracy", [])
    val_acc = h.get("val_accuracy", [])

    if not loss:
        print("Kein 'loss' im History gefunden – Plot übersprungen.")
        return

    # Epochen-Indizes und Best-Epoch (min Val-Loss) bestimmen 
    ep = np.arange(1, len(loss) + 1)        # Epochen-Indizes ab 1
    best_idx = int(np.argmin(val_loss)) if len(val_loss) else None     

    # Loss
    plt.figure()
    plt.plot(ep, loss, label="Train-Loss")
    if len(val_loss):
        plt.plot(ep, val_loss, label="Val-Loss")
        plt.axvline(best_idx + 1, linestyle="--", alpha=0.4, label="Best-Epoch (min Val-Loss)")
    plt.xlabel("Epoche"); plt.ylabel("Loss"); plt.title(f"{title} – Loss")
    plt.legend(); plt.tight_layout(); plt.show()

    # Accuracy
    if len(acc) and len(val_acc):
        plt.figure()
        plt.plot(ep, acc, label="Train-Acc")
        plt.plot(ep, val_acc, label="Val-Acc")
        if best_idx is not None:
            plt.axvline(best_idx + 1, linestyle="--", alpha=0.4)
        plt.xlabel("Epoche"); plt.ylabel("Accuracy"); plt.title(f"{title} – Accuracy")
        plt.legend(); plt.tight_layout(); plt.show()

# Funktion zur Darstellung des Generalisierungs-Gaps
# Zeigt den Unterschied zwischen Train- und Val-Accuracy über die Epochen --> Anzeichen für Overfitting
def plot_generalization_gap(history, title="Generalization Gap (Train-Acc − Val-Acc)"):
   
    # Diktionary aus History extrahieren
    h = history.history
    acc = np.array(h.get("accuracy", []), dtype=float)
    val_acc = np.array(h.get("val_accuracy", []), dtype=float)

    # Sicherstellen, dass beide Accuracy-Listen die gleiche Länge haben
    n = min(len(acc), len(val_acc))
    if n == 0:
        print("Keine Accuracy-Daten im History – Gap-Plot übersprungen.")
        return

    # Generalization Gap pro Epoche berechnen
    gap = acc[:n] - val_acc[:n]
    ep = np.arange(1, n + 1)       # Epochen-Indizes ab 1

    # Best-Epoch (min Val-Loss) bestimmen, wenn Val-Loss vorhanden
    best_idx = None
    if "val_loss" in h and len(h["val_loss"]) >= n:
        best_idx = int(np.argmin(h["val_loss"][:n]))

    plt.figure()
    plt.plot(ep, gap, label="Gap")
    if best_idx is not None:
        plt.axvline(best_idx + 1, linestyle="--", alpha=0.4, label="Best-Epoch (min Val-Loss)")
    plt.axhline(0.0, linewidth=1, color="k", alpha=0.3)
    plt.xlabel("Epoche"); plt.ylabel("Train-Acc − Val-Acc"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

# Funktion zur Pareto-Front-Berechnung
def _compute_pareto_front(results, *, x_key="parameteranzahl", y_key="accuracy", tol=1e-12):
    """
    Ergebnisse: Liste von Dicts mit Keys:
      - "parameteranzahl" (kleiner ist besser)
      - "accuracy" (größer ist besser)
    x_key/y_key zur flexiblen Wahl der Achsen (z.B. x_key="time_to_best_s", y_key="accuracy")
    Returns:
      - front_idx: Indizes der Pareto-Punkte (bezogen auf 'results')
      - front_xy:  Liste der (x, y) entlang der Front, aufsteigend nach x
    """
    # Eingangs-Arrays --> anhand derer wird die Front berechnet
    xs  = np.array([r[x_key] for r in results], dtype=float)
    ys  = np.array([r[y_key] for r in results], dtype=float)

    # Sortieren nach Parametern (aufsteigend)
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]

    # Pareto-Front bestimmen (einfacher Algorithmus, da nur 2D)
    front_mask_sorted = np.zeros(len(results), dtype=bool)
    best_y = -np.inf

    # Ein Punkt ist auf der Front, wenn es keinen Punkt mit
    # gleicher oder weniger x-Werten und besserem y gibt.
    for i, (x, y) in enumerate(zip(xs_s, ys_s)):
        if y > best_y + tol:
            front_mask_sorted[i] = True
            best_y = y

    front_idx_sorted = order[front_mask_sorted]     # Indizes auf Original-Reihenfolge mappen

    # Ergebnisse vorbereiten
    front_xy = [(xs[i], ys[i]) for i in front_idx_sorted]
    front_idx = list(front_idx_sorted[np.argsort(xs[front_idx_sorted])])
    front_xy  = sorted(front_xy, key=lambda t: t[0])

    return front_idx, front_xy      # front_idx sind die Indizes in 'results', die zur Front gehören, front_xy sind die (x, y) Paare entlang der Front

# Funktion zur Darstellung der Parameteranzahl, Inferenzzeit, Trainingszeit vs. Accuracy
def plot_params_vs_accuracy_with_front(
    results,
    title="Pareto: Parameter vs. Accuracy (Val)",
    *,
    x_key="parameteranzahl",            # variable x-Achse
    y_key="accuracy",                   # variable y-Achse (Standard: Accuracy)
    x_label=None, y_label=None,
    log_x=None,                         # automatische Log-Skala (oder explizit True/False setzen)
    show_point_legend=True,             # Standard = alle Punkte in der Legende, False = nur Pareto-Front
    keep_variant_legend=False
):
    # Markers für verschiedene Modell-Familien
    families = {r.get("family_marker", r.get("family", "Other")) for r in results}
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    marker_map = {fam: markers[i % len(markers)] for i, fam in enumerate(sorted(families))}

    fig, ax = plt.subplots()

    # Plottet alle Punkte
    for r in results:
        fam_for_marker = r.get("family_marker", r.get("family", "Other"))
        legend_label = r.get("legend_label", r.get("family", "Other"))
        ax.scatter(
            r[x_key], r[y_key],
            marker=marker_map.get(fam_for_marker, "o"),
            s=80,
            label=legend_label,
            c=r.get("color", None)
        )
        # Label direkt neben den Punkt (nur Kurzform wie "1. V.", "2. V.", …)
        ax.annotate(
            r.get("name", ""),
            (r[x_key], r[y_key]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            alpha=0.85
        )

    # Pareto-Front berechnen & einzeichnen
    front_idx, front_xy = _compute_pareto_front(results, x_key=x_key, y_key=y_key)

    if len(front_xy) >= 2:
        xs, ys = zip(*front_xy)
        ax.plot(xs, ys, linewidth=2.0, alpha=0.9,
                linestyle="-", color="black", label="Pareto-Front")
    elif len(front_xy) == 1:
        # Einzelner Frontpunkt – Linie macht keinen Sinn, aber wir markieren ihn
        x, y = front_xy[0]
        ax.scatter([x], [y], edgecolor="k", linewidth=1.5,
                   s=130, facecolors="none", label="Pareto-Punkt")

    # Achsen, Legende, Gitter
    # Standard: wie gehabt "Parameteranzahl (log)" – aber jetzt optional/automatisch
    xs_all = np.array([r[x_key] for r in results], dtype=float)
    if log_x is None:
        # Heuristik: wenn x-Spanne groß ist (z.B. Parameter/Zeit), log-Skala
        log_x = (xs_all.max() / max(xs_all.min(), 1e-12)) > 20
    if log_x:
        ax.set_xscale("log")

    if x_label is None:
        x_label = "Parameteranzahl (log)" if (x_key == "parameteranzahl" and log_x) else x_key
    if y_label is None:
        y_label = "Accuracy (Val)" if y_key == "accuracy" else y_key

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Titel ggf. automatisch anpassen, falls Keys geändert wurden
    if title is None or title == "Pareto: Parameter vs. Accuracy (Val)":
        title = f"Pareto: {y_key} vs. {x_key} (Val)"
    ax.set_title(title)

    # Familien-Duplikate in der Legende vermeiden (rückwärtskompatibel)
    handles, labels = ax.get_legend_handles_labels()

    if not show_point_legend:
        # nur Pareto-Front/-Punkt behalten
        keep = {"Pareto-Front", "Pareto-Punkt"}
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]
        if filtered:
            handles, labels = [h for h, _ in filtered], [l for _, l in filtered]
        else:
            handles, labels = [], []
    else:
        if keep_variant_legend:
            # alle Varianten in der Legende lassen (kein Deduplizieren)
            pass
        else:
            # altes Verhalten: Deduplizieren nach Label (erste Vorkommen gewinnen)
            seen = set()
            new_h, new_l = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    new_h.append(h)
                    new_l.append(l)
            handles, labels = new_h, new_l

    # Legende setzen oder entfernen, falls leer
    if handles:
        ax.legend(handles, labels, loc="best")
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    ax.grid(True, which="both", axis="both", alpha=0.15)
    fig.tight_layout()
    plt.show()




# Funktion zur Darstellung des Lernratenverlaufs
# Vor allem für Phase 5 verwendet
def plot_lr_schedule(tinfo, title="Lernratenverlauf"):
    import matplotlib.pyplot as plt
    lrs = tinfo.get("lr_per_epoch", None)
    if not lrs:
        print("Kein lr_per_epoch in tinfo gefunden (use_scheduler=False?).")
        return
    plt.figure()
    plt.plot(range(1, len(lrs)+1), lrs, marker="o")
    plt.xlabel("Epoche"); plt.ylabel("Lernrate")
    plt.title(title)
    plt.tight_layout(); plt.show()

# Funktion zur Erstellung der Confusion-Matrix
def confusion_matrix_on_val(model, x_val, y_val, *, is_cnn: bool, model_name="Model", plot=True):
    
    # Eingabedaten in richtige Form bringen
    x_eval = _to_shape(x_val, is_cnn)

    y_true = np.argmax(y_val, axis=1)       # Wahre Labels
    y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)  # Vorhergesagte Labels
    cm = confusion_matrix(y_true, y_pred)

    if plot:
        n = cm.shape[0]

        precision = precision_score(y_true, y_pred, average=None, zero_division=0) # Precision pro Klasse berechnen

        fig, ax = plt.subplots(figsize=(7.5, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(n))
        disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
        ax.set_title(f"{model_name} – Confusion Matrix")

        # Häufige Fehler hervorheben
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] >= 10:  # Schwellenwert z. B. 10
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=True, facecolor=(1, 0.6, 0.6, 0.6), edgecolor=None, linewidth=0)
                    ax.add_patch(rect)

        # Precision-Werte rechts einfügen
        ax.set_xlim(-0.5, n + 1.2)
        for i in range(n):
            ax.text(n + 0.6, i, f"{int(round(precision[i] * 100))}%",
                    ha="center", va="bottom", fontsize=10, color="darkgreen")

        plt.tight_layout()
        plt.show()

    return cm

# Funktion zur Darstellung einer Heatmap der Off-Diagonal-Fehler
def plot_offdiag_error_heatmap(model, x_val, y_val, *, is_cnn: bool, model_name="Model"):

    x_eval = _to_shape(x_val, is_cnn)
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred).astype(float)

    # Diagonale nullen und zeilenweise normalisieren
    for i in range(cm.shape[0]): cm[i, i] = 0.0
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
    cm_norm = cm / row_sums

    plt.figure(figsize=(6,4.5))
    plt.imshow(cm_norm, cmap="Reds")
    plt.colorbar(label="Fehlerrate")
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.xlabel("Vorhersage"); plt.ylabel("Wahr")
    plt.title(f"{model_name} – Verwechslungs-Heatmap")
    plt.tight_layout(); plt.show()
    return cm_norm


# Funktion zur Darstellung des Reliability-Diagramms und Berechnung des ECE
# Zeigt wie gut die vorhergesagten Wahrscheinlichkeiten kalibriert sind
def reliability_diagram_and_ece(model, x_val, y_val, *, is_cnn: bool, n_bins=10, model_name="Model"):
    x_eval = _to_shape(x_val, is_cnn)
    y_true = np.argmax(y_val, axis=1)
    probs = model.predict(x_eval, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1

    acc_bin, conf_bin, w_bin, n_bin = [], [], [], []
    N = len(conf)
    for b in range(n_bins):
        mask = (bin_ids == b)
        n_b = int(mask.sum())
        n_bin.append(n_b)
        if n_b == 0:
            acc_bin.append(0.0)
            conf_bin.append((bins[b]+bins[b+1])/2.0)
            w_bin.append(0.0)
        else:
            acc_bin.append(float(correct[mask].mean()))
            conf_bin.append(float(conf[mask].mean()))
            w_bin.append(n_b / N)

    acc_bin, conf_bin, w_bin = np.array(acc_bin), np.array(conf_bin), np.array(w_bin)
    ece = float(np.sum(w_bin * np.abs(acc_bin - conf_bin)))

    # --- Schattierung vorbereiten ---
    # normierte Gewichte 0..1 (Prozent)
    w_norm = w_bin.copy()  # bereits Anteil
    # Gamma-Stretch für stärkeren Verlauf (0.5 = „hell -> schnell dunkler“)
    gamma = 0.5
    w_shade = np.power(w_norm, gamma)
    # kleine, aber nicht-leere Bins leicht sichtbar machen
    min_tint = 0.08
    w_shade = np.where((w_norm > 0) & (w_shade < min_tint), min_tint, w_shade)

    # Colormap: Gelb -> Orange (kräftig)
    cmap = LinearSegmentedColormap.from_list("YellowOrange", ["#FFF79A", "#FF8A00"])
    norm = Normalize(vmin=0.0, vmax=1.0)  # Colorbar in 0..1

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6,5))

    # Hintergrund-Schattierung pro Bin
    for b in range(len(conf_bin)):
        color = cmap(w_shade[b]) if w_norm[b] > 0 else cmap(0.0)
        ax.fill_between([bins[b], bins[b+1]], 0, 1, color=color, zorder=0)

    # Diagonale & Modellkurve
    ax.plot([0,1], [0,1], linestyle="--", alpha=0.6, label="Perfekte Kalibrierung", color="gray")
    ax.plot(conf_bin, acc_bin, marker="o", label="Modell", color="steelblue")

    # Option: Zählungen pro Bin einblenden (oben, klein)
    for b in range(n_bins):
        if n_bin[b] > 0:
            ax.text((bins[b]+bins[b+1])/2, 1.01, str(n_bin[b]),
                    ha="center", va="bottom", fontsize=8, rotation=0, clip_on=False)

    # Colorbar (in %), ohne Rand
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Anteil Samples im Bin")
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # 0%..100%

    ax.set_xlabel("Konfidenz")
    ax.set_ylabel("Trefferrate")
    ax.set_title(f"{model_name} – Reliability (ECE={ece:.3f})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# Funktion zur Darstellung der Bias-Varianz-Balance
# Scatter-Plot: Accuracy vs. Generalisierungslücke (Train-Acc - Val-Acc)
# Punkte können nach "family" gruppiert und benannt werden
def plot_bias_variance_scatter(
    results,
    *,
    x_key="accuracy",                 # Accuracy (Val/Test) je nach Inhalt von 'results'
    y_key="generalization_gap",       # Train-Acc − Val-Acc an Best-Epoch (positiv = eher overfit)
    name_key="name",                  # Label für einzelne Punkte (z. B. "1", "2", …)
    family_key="family",              # Gruppierung/Farbe
    title="Bias–Varianz Balance: Accuracy vs. Generalisierungslücke",
    x_label="Accuracy",
    y_label="Generalisierungslücke (Train−Val)",
    highlight_names=None,             # Liste von Namen, die hervorgehoben werden sollen (z. B. ["8","9"])
    family_colors=None,               # Optional dict: {family: color}
    point_size=80,
    annotate=True,                    # Punkt-Beschriftung neben Marker
):
    """
    Erwartetes 'results'-Format: Liste von Dicts mit mindestens
      { "name": str, "family": str, "accuracy": float, "generalization_gap": float }
    Optional weitere Keys sind egal.
    """

    # Farben je Family – stabiler Default
    default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    families = sorted({r.get(family_key, "Other") for r in results})
    if family_colors is None:
        family_colors = {fam: default_palette[i % len(default_palette)] for i, fam in enumerate(families)}
    else:
        # Fehlende Familienfarben mit Defaults auffüllen
        for i, fam in enumerate(families):
            family_colors.setdefault(fam, default_palette[i % len(default_palette)])

    fig, ax = plt.subplots()
    handles = {}
    all_x, all_y = [], []

    # Scatter pro Punkt
    for r in results:
        fam = r.get(family_key, "Other")
        c = family_colors.get(fam)
        x = float(r.get(x_key, np.nan))
        y = float(r.get(y_key, np.nan))
        if np.isnan(x) or np.isnan(y):
            continue
        all_x.append(x); all_y.append(y)

        sc = ax.scatter(x, y, s=point_size, color=c, alpha=0.9, edgecolor="white", linewidth=0.6)
        # Sammle je Family einen Handle für die Legende
        if fam not in handles:
            handles[fam] = sc

        # dezentes Label neben den Punkt
        if annotate:
            ax.annotate(str(r.get(name_key, "")), (x, y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, alpha=0.85)

    if not all_x:
        print("Keine gültigen Daten für Plot gefunden.")
        return

    all_x = np.array(all_x); all_y = np.array(all_y)

    # Hilfslinien: Median-Accuracy & Median-Gap (orientierung)
    x_med = float(np.median(all_x))
    y_med = float(np.median(all_y))
    ax.axvline(x_med, color="gray", linestyle="--", alpha=0.35, linewidth=1)
    ax.axhline(y_med, color="gray", linestyle="--", alpha=0.35, linewidth=1)

    # Null-Linie für Gap (Train−Val = 0 → perfekt kalibriert)
    ax.axhline(0.0, color="black", linestyle=":", alpha=0.4, linewidth=1)

    # Optionale Highlights (z. B. Varianten "8", "9")
    if highlight_names:
        for r in results:
            if str(r.get(name_key)) in set(map(str, highlight_names)):
                x = float(r.get(x_key, np.nan))
                y = float(r.get(y_key, np.nan))
                if np.isnan(x) or np.isnan(y):
                    continue
                ax.scatter(x, y, s=point_size*1.3, facecolors="none", edgecolors="black", linewidths=1.6, zorder=5)
                # leichte, sichtbare Beschriftung zusätzlich
                ax.annotate(f"★ {r.get(name_key)}", (x, y),
                            xytext=(8, 10), textcoords="offset points",
                            fontsize=10, fontweight="bold", color="black")

    # Achsenbeschriftung & Titel
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Legende: alle Familien + Erklärung „Pareto-Front analog: rechts unten besser (stabil)“
    # -> Wir bauen die Legende aus den Family-Handles
    fam_handles = list(handles.values())
    fam_labels  = list(handles.keys())
    if fam_handles:
        ax.legend(fam_handles, fam_labels, title="Gruppen", loc="best", frameon=True)

    # dezentes Grid
    ax.grid(True, which="both", axis="both", alpha=0.15)

    # Grenzen mit etwas Padding
    pad_x = max(0.001, 0.02 * (all_x.max() - all_x.min() if all_x.max() > all_x.min() else 1))
    pad_y = max(0.001, 0.15 * (all_y.max() - all_y.min() if all_y.max() > all_y.min() else 1))
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    fig.tight_layout()
    plt.show()


# Funktion zur Untersuchung der Robustheit gegen Rauschen
# Fügt dem Validierungsset gaußsches Rauschen hinzu und misst die Accuracy
# Jedem Pixel wird über eine Normalverteilung N(0, sigma) ein Rauschwert hinzugefügt
def accuracy_vs_noise(model, x_val, y_val, *, is_cnn: bool, sigmas=(0.0, 0.05, 0.10, 0.15), model_name="Model"):
    x_base = x_val.astype("float32")
    res = []
    for s in sigmas:
        if s == 0.0:
            x_noisy = x_base
        else:
            noise = np.random.normal(0.0, s, size=x_base.shape).astype("float32")
            x_noisy = np.clip(x_base + noise, 0.0, 1.0)
        x_eval = _to_shape(x_noisy, is_cnn)
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)
        acc = float(np.mean(y_true == y_pred))
        res.append(acc)

    plt.figure()
    plt.plot(sigmas, res, marker="o")
    plt.xlabel("Rauschstärke σ"); plt.ylabel("Accuracy (Val)")
    plt.title(f"{model_name} – Robustheit gegen Rauschen")
    plt.tight_layout(); plt.show()
    return dict(zip(sigmas, res))



# Funktion zur Anzeige eines verrauschten Beispiels
# Zeigt das Originalbild und die verrauschte Version nebeneinander
def show_noisy_example(x_data, sigma=0.1, idx=0):

    original = x_data[idx]

    # Gaußsches Rauschen hinzufügen
    noise = np.random.normal(0.0, sigma, size=original.shape).astype("float32")
    noisy = np.clip(original + noise, 0.0, 1.0)

    # Darstellung
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title(f"Rauschen σ={sigma}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return noisy
