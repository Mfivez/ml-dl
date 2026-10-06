"""Schémas exacts du même cycle, de la lecture à l'apprentissage."""

import numpy as np
import matplotlib.pyplot as plt
from j4_visual_additions import _canvas, _text, _box, _arrow, BLUE, ORANGE, GREEN, PURPLE, MUTED


def build_progression_visuals(save):
    cycle = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])
    inverse = cycle[[0, 2, 1, 3]]
    memoire, trace = 0.0, []
    for temperature, vibration in cycle:
        memoire = float(np.tanh(temperature - vibration + 0.6 * memoire))
        trace.append(memoire)
    scores = [1 / (1 + np.exp(-poids * memoire)) for poids in [4, -4]]

    fig, ax = _canvas("Le même cycle : des mesures à la mémoire, puis à une réponse", size=(13, 7.5), ylim=(0, 7.5))
    _text(ax, 6.5, 7.12, "Classe attendue : 0 = vibration puis température", size=15, weight="bold", color=BLUE)
    _text(ax, 6.5, 6.55, "Mémoire initiale = 0 ; nouvelle mémoire = tanh(T − V + 0,6 × ancienne mémoire)", size=12)
    labels = ["Repos", "Vibration", "Température", "Repos"]
    for i, (row, label, state) in enumerate(zip(cycle, labels, trace)):
        x = .25 + 3.2 * i
        _box(ax, x, 3.83, 2.85, 2.05, color=BLUE)
        _text(ax, x + 1.425, 5.50, f"Instant {i} : {label}", size=12, weight="bold")
        _text(ax, x + 1.425, 4.93, f"T = {row[0]}    V = {row[1]}", size=14, color=BLUE)
        _text(ax, x + 1.425, 4.32, f"Mémoire : {state:.3f}".replace(".", ","), size=14, color=PURPLE, weight="bold")
        if i < 3:
            _arrow(ax, (x + 2.90, 4.35), (x + 3.12, 4.35), PURPLE)
    _text(ax, 6.5, 3.29, "Au dernier repos, les capteurs valent 0 mais la mémoire garde une trace du parcours.", size=12)
    for x, weight, score, verdict, color in [(.3, 4, scores[0], "classe 1 : réponse fausse", ORANGE),
                                            (6.8, -4, scores[1], "classe 0 : réponse juste", GREEN)]:
        _box(ax, x, 1.20, 5.9, 1.53, color=color)
        _text(ax, x + 2.95, 2.36, f"Même mémoire, poids de sortie = {weight:+d}", size=12, weight="bold", color=color)
        _text(ax, x + 2.95, 1.93, f"sigmoid({weight:+d} × {memoire:.3f}) ≈ {score:.3f}".replace(".", ","), size=13)
        _text(ax, x + 2.95, 1.52, verdict + " (seuil 0,5)", size=12)
    _text(ax, 6.5, .70, "Deux réglages choisis À LA MAIN pour comprendre ; ce n’est pas encore un entraînement.", size=12, weight="bold")
    _text(ax, 6.5, .22, "T et V : événements absents (0) ou présents (1), pas des unités physiques. Valeurs affichées arrondies.", size=11, color=MUTED)
    save(fig, "00_cycle_memoire_score")

    fig, ax = _canvas("Lire n’est pas apprendre : deux opérations différentes", size=(13, 6.5), ylim=(0, 6.5))
    for x, title, color in [(1.6, "Une séquence\nmémoire remise à 0", BLUE), (6.3, "Lire les instants\nmettre à jour la mémoire", PURPLE), (11, "Résumé final\npuis score", GREEN)]:
        _box(ax, x - 1.55, 4.45, 3.1, 1.22, color=color, label=title, size=12)
    _arrow(ax, (3.2, 5.05), (4.69, 5.05)); _arrow(ax, (7.9, 5.05), (9.4, 5.05))
    _text(ax, 6.5, 6.14, "Pendant une lecture : les poids restent fixes, la mémoire change.", size=13, weight="bold")
    _box(ax, 8.9, 2.12, 3.8, 1.2, color=ORANGE, label="Scores du lot + bonnes classes\n→ perte à réduire", size=12)
    _arrow(ax, (11, 4.37), (11, 3.43), ORANGE)
    _box(ax, .25, 2.12, 6.5, 1.2, color=ORANGE, label="L’optimiseur ajuste les poids\nde la mémoire ET de la sortie", size=13)
    _arrow(ax, (8.8, 2.72), (6.88, 2.72), ORANGE)
    _arrow(ax, (1.6, 3.42), (1.6, 4.36), ORANGE)
    _text(ax, 6.5, 1.51, "On recommence sur les exemples d’entraînement avec les poids ajustés.", size=13)
    _text(ax, 6.5, .80, "Validation et test : lire et mesurer les erreurs, sans corriger les poids.", size=13, color=GREEN, weight="bold")
    _text(ax, 6.5, .27, "Un lot regroupe plusieurs séquences : chacune commence avec sa propre mémoire à zéro.", size=11, color=MUTED)
    save(fig, "00_deux_boucles")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, (ax, seq) in enumerate(zip(axes, [cycle, inverse])):
        ax.axis("off")
        table = ax.table(cellText=seq, colLabels=["Température T", "Vibration V"],
                         rowLabels=["instant 0", "instant 1", "instant 2", "instant 3"],
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2)
        ax.set_title(f"X[{i}] : 4 instants × 2 capteurs\nClasse {i}", pad=10)
    fig.suptitle("Les mêmes deux cycles, rangés sans perdre leur ordre : X.shape = (2, 4, 2)", fontsize=15)
    fig.text(.5, .08, "X[0, 2, 0] = 1 : cycle 0, instant 2, température présente", ha="center", fontsize=13, color=ORANGE)
    fig.subplots_adjust(left=.11, right=.97, wspace=.57, top=.75, bottom=.20)
    save(fig, "00_tenseur_3d")

    fig, ax = _canvas("Le RNN déplié : une seule règle, utilisée quatre fois", size=(13, 5), ylim=(0, 5))
    for i, label in enumerate(labels):
        x = .3 + 3.2 * i
        _box(ax, x, 2.0, 2.65, 1.03, color=ORANGE, label=f"Même cellule\ninstant {i}", size=12)
        _text(ax, x + 1.325, .71, f"{label}\nT = {cycle[i, 0]}, V = {cycle[i, 1]}", size=12, color=BLUE)
        _arrow(ax, (x + 1.325, 1.25), (x + 1.325, 1.93), BLUE)
        if i < 3:
            _arrow(ax, (x + 2.7, 2.51), (x + 3.14, 2.51), PURPLE)
            _text(ax, x + 2.93, 3.23, "mémoire", size=10, color=PURPLE)
    _text(ax, 1.625, 4.4, "Départ : mémoire = 0", size=12, color=PURPLE)
    _arrow(ax, (1.625, 4.04), (1.625, 3.1), PURPLE)
    _arrow(ax, (11.225, 3.1), (11.225, 4.04), GREEN)
    _text(ax, 11.225, 4.4, "Résumé → score", size=12, color=GREEN)
    _text(ax, 6.5, .08, "Il ne s’agit pas de quatre réseaux : les mêmes poids sont partagés entre les instants.", size=12, weight="bold")
    save(fig, "00_rnn_deplie")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    rows = [["1 · Lire", "Identifier les séquences, les instants et les capteurs"],
            ["2 · Prédire", "Dire ce qui change si les événements sont échangés"],
            ["3 · Préparer", "Garder les jumeaux ensemble ; utiliser les statistiques du train"],
            ["4 · Construire", "Relier entrée → mémoire → score"],
            ["5 · Apprendre", "Ajuster sur le train ; surveiller la validation"],
            ["6 · Comparer", "Observer une paire de validation et les poids inchangés"],
            ["7 · Évaluer", "Ouvrir le test une seule fois, les réglages étant fixés"],
            ["8 · Expliquer", "Distinguer mécanisme appris et limites de l’expérience"]]
    table = ax.table(cellText=rows, colLabels=["Votre parcours", "Ce qu’il faut pouvoir expliquer"],
                     colWidths=[.20, .80], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2.5)
    ax.set_title("Reconstruire l’expérience en comprenant chaque choix", fontsize=16, pad=22)
    save(fig, "02_parcours_exercice")
