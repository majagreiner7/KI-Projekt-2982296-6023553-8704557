# Einstellungen für vergleichbare Ergebnisse indem determinisitsche Einstellungen erzwungen werden
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import random, numpy as np
import tensorflow as tf

# Versuche, deterministische Operationen zu erzwingen (funktioniert nicht immer, wenn nicht wird einfach weitergemacht)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from model_train import set_seed, prepare_data, build_cnn_basic, build_mlp_basic, build_cnn_regularized, compile_model, train, evaluate_on_val
from plots import plot_learning_curves, plot_generalization_gap, plot_lr_schedule, confusion_matrix_on_val, plot_offdiag_error_heatmap, reliability_diagram_and_ece, accuracy_vs_noise, plot_params_vs_accuracy_with_front, show_noisy_example, plot_bias_variance_scatter


if __name__ == "__main__":

    # Seed setzen für Reproduzierbarkeit
    set_seed(0)

    # alles in einer Funktion: Laden, Normalisieren, Split, One-Hot
    x_tr, y_tr, x_val, y_val, x_test, y_test = prepare_data(test_size=0.1, seed=0)

    """
    # Phase 1: MLP Modell ausführen 
    mlp = build_mlp_basic()
    compile_model(mlp, optimizer="adam", lr=1e-3)
    h_mlp, tinfo_mlp = train(mlp, x_tr, y_tr, x_val, y_val, is_cnn=False, batch_size=64, epochs=20, patience=3, verbose=1, seed=0)
    m_mlp = evaluate_on_val(mlp, h_mlp, x_val, y_val, is_cnn=False)
    m_mlp.update(tinfo_mlp)

    print("=== MLP (Validation) ===")
    for k,v in m_mlp.items():
        if isinstance(v, float):
            print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

    #plot_learning_curves(h_mlp, title="MLP – Lernkurven (Phase 1)")
    #plot_generalization_gap(h_mlp, title="MLP – Generalization Gap (Phase 1)")
    confusion_matrix_on_val(mlp, x_val, y_val, is_cnn=False, plot=True, model_name="MLP")
    plot_offdiag_error_heatmap(mlp, x_val, y_val, is_cnn=False, model_name="MLP – Fehler-Heatmap (Phase 1)")
    reliability_diagram_and_ece(mlp, x_val, y_val, is_cnn=False, n_bins=10, model_name="MLP - Phase 1")
    _= accuracy_vs_noise(mlp, x_val, y_val, is_cnn=False, model_name="MLP - Phase 1")

    
    plot_learning_curves(h_mlp, title="MLP – Lernkurven (Phase 1)")
    plot_generalization_gap(h_mlp, title="MLP – Generalization Gap (Phase 1)")

    # Phase 1+2: CNN Modell ausführen
    # epochs=20, patience=3
    cnn_1_2 = build_cnn_basic()
    compile_model(cnn_1_2, optimizer="adam", lr=1e-3)
    h_cnn_1_2, tinfo_cnn_1_2 = train(cnn_1_2, x_tr, y_tr, x_val, y_val, is_cnn=True, verbose=1, seed=0, use_scheduler=False, batch_size=64, epochs=20, patience=3)
    m_cnn_1_2 = evaluate_on_val(cnn_1_2, h_cnn_1_2, x_val, y_val, is_cnn=True)
    m_cnn_1_2.update(tinfo_cnn_1_2)

    print("=== CNN-Phase 1/2 (Validation) ===")
    for k,v in m_cnn_1_2.items():
        if isinstance(v, float):
            print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

    plot_learning_curves(h_cnn_1_2, title="CNN – Lernkurven (Phase 1/2)")
    plot_generalization_gap(h_cnn_1_2, title="CNN – Generalization Gap (Phase 1/2)")


    # Phase 3: CNN Modell ausführen
    # epochs=60, patience=5, batch_size=128 (bei Verwendung von BatchNormalization, sonst 64)
    cnn_3 = build_cnn_regularized()
    compile_model(cnn_3, optimizer="adam", lr=1e-3)
    h_cnn_3, tinfo_cnn_3 = train(cnn_3, x_tr, y_tr, x_val, y_val, is_cnn=True, verbose=1, seed=0, use_scheduler=True, batch_size=128, epochs=60, patience=5)
    m_cnn_3 = evaluate_on_val(cnn_3, h_cnn_3, x_val, y_val, is_cnn=True)
    m_cnn_3.update(tinfo_cnn_3)

    print("=== CNN-Phase 3 (Validation) ===")
    for k,v in m_cnn_3.items():
        if isinstance(v, float):
            print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")
    
    # Phase 4: CNN Modell ausführen
    # Phase 4: batch_size=256, epochs=60, patience=5
    cnn_4 = build_cnn_regularized()
    compile_model(cnn_4, optimizer="adam", lr=1e-3)
    h_cnn_4, tinfo_cnn_4 = train(cnn_4, x_tr, y_tr, x_val, y_val, is_cnn=True, verbose=1, seed=0, use_scheduler=True, batch_size=256, epochs=60, patience=5)
    m_cnn_4 = evaluate_on_val(cnn_4, h_cnn_4, x_val, y_val, is_cnn=True)
    m_cnn_4.update(tinfo_cnn_4)

    print("=== CNN-Phase 4 (Validation) ===")
    for k,v in m_cnn_4.items():
        if isinstance(v, float):
            print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")


    
    # Phase 5: CNN Modell ausführen
    # Phase 5: batch_size=256, epochs=60, patience=5
    cnn_5 = build_cnn_regularized()
    compile_model(cnn_5, optimizer="adam", lr=1e-3)
    h_cnn_5, tinfo_cnn_5 = train(cnn_5, x_tr, y_tr, x_val, y_val, is_cnn=True, verbose=1, seed=0, use_scheduler=True, batch_size=256, epochs=60, patience=5)
    m_cnn_5 = evaluate_on_val(cnn_5, h_cnn_5, x_val, y_val, is_cnn=True)
    m_cnn_5.update(tinfo_cnn_5)

    print("=== CNN-Phase 5 (Validation) ===")
    for k,v in m_cnn_5.items():
        if isinstance(v, float):
            print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

    plot_lr_schedule(tinfo_cnn_5, title="CNN – Lernratenverlauf (Phase 5)")
    
    """

    """
    # Für Durchläufe mit verschiedenen Seeds (z.B. 5) um statistische Schwankungen zu minimieren
    seeds = [0, 1, 2, 3, 4]
    histories, metrics = [], []

    for s in seeds:
        tf.keras.backend.clear_session()
        set_seed(s)
        cnn = build_cnn_regularized()                        
        compile_model(cnn, optimizer="sgd", lr=1e-2)
        h, tinfo = train(cnn, x_tr, y_tr, x_val, y_val, is_cnn=True, verbose=0, seed=s, use_scheduler=True)
        m = evaluate_on_val(cnn, h, x_val, y_val, is_cnn=True)
        m.update(tinfo)
        histories.append(h); metrics.append(m)

    # Mittelwert-Print
    acc = np.array([m["accuracy"] for m in metrics], float)
    f1  = np.array([m["f1"] for m in metrics], float)
    par = np.array([m["parameteranzahl"] for m in metrics], float)
    inf = np.array([m["inferenzzeit_ms_pro_beispiel"] for m in metrics], float)
    trt = np.array([m["train_time_s"] for m in metrics], float)
    gap = np.array([m["generalization_gap"] for m in metrics], float)
    ttb = np.array([m["time_to_best_s"] for m in metrics], float)
    bep = np.array([m["best_epoch"] for m in metrics], float)

    print(f"Accuracy            : {acc.mean():.5f} ± {acc.std():.5f}")
    print(f"F1 (macro)          : {f1.mean():.5f} ± {f1.std():.5f}")
    print(f"Parameteranzahl     : {par.mean():.1f} ± {par.std():.1f}")
    print(f"Inferenz (ms/Bsp)   : {inf.mean():.3f} ± {inf.std():.3f}")
    print(f"Trainingszeit [s]   : {trt.mean():.1f} ± {trt.std():.1f}")
    print(f"Generalization Gap  : {gap.mean():+.5f} ± {gap.std():.5f}")
    print(f"Time→Best [s]       : {ttb.mean():.1f} ± {ttb.std():.1f}")
    print(f"Beste Epoche        : {bep.mean():.1f} ± {bep.std():.1f}")


    # Zum Plotten wird der Mittelwert aus den Runs verwendet)
    mid = int(np.argsort(acc)[len(acc) // 2]) 
    #plot_learning_curves(histories[mid], title="Learning Curves – Keine Regularisierung")
    #plot_generalization_gap(histories[mid], title="Generalization Gap – Keine Regularisierung")
    """


"""
    # Modell-Spezifikationen (finale Settings)
    specs = [
        dict(name="MLP (Basis)", family="MLP",
             build_fn=build_mlp_basic, is_cnn=False,
             batch_size=64,  epochs=60, patience=5,
             optimizer="adam", lr=1e-3, use_scheduler=False),

        dict(name="CNN (Basis)", family="CNN",
             build_fn=build_cnn_basic, is_cnn=True,
             batch_size=64,  epochs=60, patience=5,
             optimizer="adam", lr=1e-3, use_scheduler=False),

        dict(name="CNN (Optimiert)", family="CNN",
             build_fn=build_cnn_regularized, is_cnn=True,
             batch_size=256, epochs=60, patience=5,
             optimizer="adam", lr=1e-3, use_scheduler=True),
    ]

    results_for_pareto = []
    all_metrics = []

    for sp in specs:
        # Frischer Graph + Seed
        tf.keras.backend.clear_session()
        set_seed(0)

        # 1 Modell bauen & kompilieren
        model = sp["build_fn"]()
        compile_model(model, optimizer=sp["optimizer"], lr=sp["lr"])

        # 2 TRAINING: nur Train + Val (Val für EarlyStopping & Restore Best Weights)
        history, tinfo = train(
            model, x_tr, y_tr, x_val, y_val,
            is_cnn=sp["is_cnn"],
            batch_size=sp["batch_size"], epochs=sp["epochs"], patience=sp["patience"],
            verbose=1, seed=0, use_scheduler=sp["use_scheduler"]
        )

        # Trainingsdiagnostik (optional—nutzt History, kein Testzugriff)
        plot_learning_curves(history, title=f"{sp['name']} – Lernkurven")
        plot_generalization_gap(history, title=f"{sp['name']} – Generalization Gap")

        # 3 Finale Analyse ausschließlich auf Testdaten
        title_test = f"{sp['name']} – Test"
        confusion_matrix_on_val(model, x_test, y_test, is_cnn=sp["is_cnn"], plot=True, model_name=title_test)
        plot_offdiag_error_heatmap(model, x_test, y_test, is_cnn=sp["is_cnn"], model_name=title_test)
        plot_accuracy_vs_topk(model, x_test, y_test, is_cnn=sp["is_cnn"], k=2, model_name=title_test)
        reliability_diagram_and_ece(model, x_test, y_test, is_cnn=sp["is_cnn"], n_bins=10, model_name=title_test)
        _ = accuracy_vs_noise(model, x_test, y_test, is_cnn=sp["is_cnn"],
                              sigmas=(0.0, 0.05, 0.10, 0.15), model_name=title_test)

        # 4 TEST-Kennzahlen berechnen (Funktion ist generisch—wir geben TEST rein)
        m = evaluate_on_val(model, history, x_test, y_test, is_cnn=sp["is_cnn"])
        m.update(tinfo)
        m["name"] = sp["name"]; m["family"] = sp["family"]
        all_metrics.append(m)

        # Robuste Mappings (akzeptiert englische oder deutsche Keys)
        param_count = m.get("parameteranzahl", m.get("number_of_params"))
        infer_ms    = m.get("inferenzzeit_ms_pro_beispiel", m.get("inference_time_ms_for_example"))

        # Konsole: vollständige TEST-Metriken
        print(f"\n=== {sp['name']} (TEST) – alle Metriken ===")
        for key in sorted(m.keys()):
            val = m[key]
            print(f"{key}: {val:.5f}" if isinstance(val, float) else f"{key}: {val}")

        results_for_pareto.append({
            "name": sp["name"],
            "family": sp["family"],
            "parameteranzahl": float(param_count),
            "accuracy": float(m["accuracy"]),
            "inferenzzeit_ms_pro_beispiel": float(infer_ms),
            "time_to_best_s": float(m["time_to_best_s"]),
        })

    # 5 Drei Pareto-Plots (Accuracy vs. Parameter | Inferenzzeit | Time-to-Best) – TEST
    plot_params_vs_accuracy_with_front(
        results_for_pareto,
        x_key="parameteranzahl", y_key="accuracy",
        x_label="Parameteranzahl (log)", y_label="Accuracy (Test)",
        title="Pareto: Accuracy vs. Parameter (Test)"
    )
    plot_params_vs_accuracy_with_front(
        results_for_pareto,
        x_key="inferenzzeit_ms_pro_beispiel", y_key="accuracy",
        x_label="Inferenzzeit pro Beispiel [ms]", y_label="Accuracy (Test)",
        title="Pareto: Accuracy vs. Inferenzzeit (Test)",
        log_x=False
    )
    plot_params_vs_accuracy_with_front(
        results_for_pareto,
        x_key="time_to_best_s", y_key="accuracy",
        x_label="Time-to-Best [s]", y_label="Accuracy (Test)",
        title="Pareto: Accuracy vs. Time-to-Best (Test)",
        log_x=False
    )

   """