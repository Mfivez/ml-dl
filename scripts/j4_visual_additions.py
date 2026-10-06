"""Schémas du jour 4 : calculs contrôlés et formes expliquées visuellement.

Le générateur principal appelle build_visual_additions(save).
save(fig, name) fournit les exports PNG et SVG du cours.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


BLUE, ORANGE, GREEN, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#9333ea"
INK, MUTED, PALE = "#0f172a", "#475569", "#f1f5f9"


def _canvas(title, size=(13, 6), xlim=(0, 13), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=18, color=INK)
    return fig, ax


def _text(ax, x, y, text, *, size=12, color=INK, weight="normal", ha="center", va="center"):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.5)


def _box(ax, x, y, width, height, *, color=BLUE, face="white", label=None, size=12):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.035,rounding_size=0.10",
                          linewidth=1.7, edgecolor=color, facecolor=face)
    ax.add_patch(patch)
    if label is not None:
        _text(ax, x + width / 2, y + height / 2, label, size=size)
    return patch


def _arrow(ax, start, end, color=MUTED, *, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                linewidth=1.7, color=color,
                                linestyle="--" if dashed else "-"))


def _memory(save):
    fig, ax = _canvas("Une mémoire qui se met à jour, mesure après mesure", size=(13, 5.7))
    _text(ax, 6.5, 5.65, "Règle : nouvelle mémoire = tanh(mesure + 0,6 × ancienne mémoire)", size=14)
    _text(ax, 6.5, 5.05, "Au départ : mémoire = 0. Les nombres 1 et 0,6 sont fixés pour cet exemple.", color=MUTED)
    measures = [0.2, 0.8, 0.4]
    previous = 0.0
    expected = [0.197, 0.725, 0.683]
    for index, (x, measure) in enumerate(zip([0.25, 4.55, 8.85], measures)):
        current_sum = measure + 0.6 * previous
        current = float(np.tanh(current_sum))
        assert abs(current - expected[index]) < 0.0005
        fmt = lambda value: f"{value:.3f}".replace(".", ",")
        _box(ax, x, 1.2, 3.8, 3.25, color=PURPLE)
        _text(ax, x + 1.9, 4.05, f"Instant {index + 1}", color=PURPLE, weight="bold", size=14)
        _text(ax, x + 1.9, 3.5, f"Mesure lue : {str(measure).replace('.', ',')}", color=BLUE, weight="bold")
        _text(ax, x + 1.9, 2.96, f"Ancienne mémoire : {fmt(previous)}")
        _text(ax, x + 1.9, 2.40, f"Somme avant tanh : {fmt(current_sum)}")
        _box(ax, x + .22, 1.42, 3.36, .62, color=PURPLE, face="#faf5ff",
             label=f"Nouvelle mémoire : {fmt(current)}", size=12)
        if index < 2:
            _arrow(ax, (x + 3.85, 2.8), (x + 4.2, 2.8), PURPLE)
        previous = current
    _text(ax, 6.5, .66, "La nouvelle mémoire d’une étape devient l’ancienne mémoire de la suivante.", weight="bold")
    _text(ax, 6.5, .13, "Affichage arrondi à 3 décimales ; calculs non arrondis. Ici : lecture, pas apprentissage.", size=11, color=MUTED)
    save(fig, "00_memoire_calcul")


def _tanh(save):
    fig, (ax, note) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.4, 1]})
    values = np.linspace(-3, 3, 301)
    ax.plot(values, np.tanh(values), color=PURPLE, linewidth=3)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(-1, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(0, color="#cbd5e1", linewidth=1)
    ax.axvline(0, color="#cbd5e1", linewidth=1)
    ax.scatter([.2], [np.tanh(.2)], color=ORANGE, s=65, zorder=4)
    ax.annotate("Entrée 0,2\nSortie ≈ 0,197", xy=(.2, np.tanh(.2)), xytext=(.95, -.5),
                fontsize=12, ha="left", va="center", color=INK,
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5})
    ax.set(xlim=(-3.1, 3.1), ylim=(-1.25, 1.25), xticks=np.arange(-3, 4), yticks=[-1, 0, 1],
           xlabel="Nombre avant tanh", ylabel="Nombre après tanh")
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=.12)
    note.set(xlim=(0, 5), ylim=(0, 5)); note.axis("off")
    _box(note, .2, 3.58, 4.6, 1, color=PURPLE, face="#faf5ff",
         label="tanh borne la mémoire\nentre −1 et +1.", size=13)
    _text(note, 2.5, 2.55, "Une valeur négative\nest donc possible.", size=13)
    _text(note, 2.5, 1.15, "Ce n’est pas une probabilité :\nla mémoire sert de résumé,\npas de pourcentage de confiance.", size=12)
    fig.suptitle("tanh : une transformation qui limite la valeur de la mémoire", fontsize=16, weight="bold", y=1.02)
    fig.subplots_adjust(wspace=.25, bottom=.16, top=.90)
    save(fig, "00_tanh")


def _score(save):
    fig, ax = _canvas("Du résumé d’un cycle à une décision", size=(13, 5.4), ylim=(0, 5.4))
    _box(ax, .2, 3.65, 3.6, .98, color=PURPLE, label="Dernière mémoire\n= résumé numérique")
    _arrow(ax, (3.87, 4.14), (4.65, 4.14))
    _box(ax, 4.72, 3.65, 3.2, .98, color=GREEN, label="Un score entre 0 et 1\npour la classe 1")
    _text(ax, 10.42, 4.13, "Puis une règle :\nscore ≥ 0,5 → classe 1", size=12, weight="bold")
    left, right, threshold = .9, 12.1, 6.5
    ax.plot([left, threshold], [2.68, 2.68], color=BLUE, linewidth=8, solid_capstyle="butt")
    ax.plot([threshold, right], [2.68, 2.68], color=ORANGE, linewidth=8, solid_capstyle="butt")
    for value, x in [(0, left), (.5, threshold), (1, right)]:
        ax.plot([x, x], [2.5, 2.86], color=INK, linewidth=1.5)
        _text(ax, x, 3.14, str(value).replace(".", ","), size=12)
    for value, x, color, label in [
        (.2, left + .2 * (right-left), BLUE, "0,2 → classe 0\nVibration, puis température"),
        (.8, left + .8 * (right-left), ORANGE, "0,8 → classe 1\nTempérature, puis vibration"),
    ]:
        ax.scatter([x], [2.68], s=130, color=color, edgecolor="white", linewidth=2, zorder=4)
        _text(ax, x, 1.86, label, color=color, weight="bold", size=12)
    _text(ax, threshold, .93, "À 0,5 exactement : classe 1 avec cette règle.", size=12)
    _text(ax, threshold, .27, "Scores fictifs pour lire le seuil. Un score élevé n’est pas une certitude.", color=MUTED, size=11)
    save(fig, "00_score_decision")


def _groups(save):
    fig, ax = _canvas("Séparer les cycles entiers : les deux jumeaux restent ensemble", size=(13, 6.3), ylim=(0, 6.3))
    _text(ax, 6.5, 5.98, "400 cycles × 2 versions du même cycle = 800 séquences", size=14, weight="bold")
    _text(ax, 6.5, 5.41, "On répartit les numéros de cycle, puis on récupère leurs deux séquences.", color=MUTED)
    columns = [(.15, BLUE, "ENTRAÎNEMENT", "256 cycles → 512 séquences", "64 %", 42, "Apprendre les poids"),
               (4.51, ORANGE, "VALIDATION", "64 cycles → 128 séquences", "16 %", 73, "Surveiller l’apprentissage"),
               (8.87, GREEN, "TEST", "80 cycles → 160 séquences", "20 %", 218, "Faire le bilan final")]
    for x, color, title, counts, percentage, group, role in columns:
        _box(ax, x, 1.18, 3.98, 3.8, color=color)
        center = x + 1.99
        _text(ax, center, 4.58, title, color=color, weight="bold", size=13)
        _text(ax, center, 4.07, counts, size=11.5)
        _text(ax, center, 3.61, percentage, size=17, color=color, weight="bold")
        _box(ax, x + .23, 2.05, 3.52, 1.15, color=color, face=PALE)
        _text(ax, center, 2.87, f"Numéro de cycle : {group}", size=11.5, weight="bold")
        _text(ax, center, 2.40, "Jumeau classe 0 + jumeau classe 1", size=11)
        _text(ax, center, 1.6, role, size=12, color=color, weight="bold")
    _text(ax, 6.5, .67, "Un même numéro de cycle n’appartient jamais à deux ensembles.", weight="bold", size=12)
    _text(ax, 6.5, .17, "Les numéros 42, 73 et 218 illustrent la règle ; ils ne décrivent pas le tirage réel.", color=MUTED, size=11)
    save(fig, "01_groupes_decoupage")


def _standardisation(save):
    example = np.array([[40., 1.], [50., 2.], [60., 3.]])
    np.testing.assert_allclose(example.mean(axis=0), [50., 2.])
    np.testing.assert_allclose(example.std(axis=0), [8.165, .8165], atol=.00004)
    np.testing.assert_allclose((example - example.mean(axis=0)) / example.std(axis=0),
                               [[-1.225, -1.225], [0., 0.], [1.225, 1.225]], atol=.0003)
    fig, ax = _canvas("Standardiser : une moyenne et un écart-type par capteur", size=(13, 8.5), ylim=(0, 8.5))
    _text(ax, 6.5, 8.14, "Petit exemple indépendant : trois mesures de température et de vibration", size=13, color=MUTED)
    # Le tableau est dessiné explicitement pour contrôler l'espacement des textes.
    def table(x, title, rows, headings, color, width=3.35):
        _text(ax, x + width/2, 7.5, title, size=13, color=color, weight="bold")
        _box(ax, x, 4.75, width, 2.37, color=color)
        ax.plot([x + width/2, x + width/2], [4.75, 7.12], color="#cbd5e1", linewidth=1)
        for index, heading in enumerate(headings):
            _text(ax, x + width * (.25 + .5*index), 6.77, heading, size=11)
        for row, values in enumerate(rows):
            y = 6.16 - row * .5
            for index, value in enumerate(values):
                _text(ax, x + width * (.25 + .5*index), y, value, size=13)
        ax.plot([x, x + width], [6.43, 6.43], color="#cbd5e1", linewidth=1)
    table(.15, "Mesures brutes", [["40", "1"], ["50", "2"], ["60", "3"]],
          ["Température\n°C", "Vibration\nmm/s"], BLUE)
    _text(ax, 6.5, 7.48, "Statistiques de cet exemple", size=12, color=PURPLE, weight="bold")
    _box(ax, 4.13, 5.04, 4.74, 2.09, color=PURPLE, face="#faf5ff")
    _text(ax, 6.38, 6.74, "Température", size=11, weight="bold")
    _text(ax, 8.04, 6.74, "Vibration", size=11, weight="bold")
    for y, label, temperature, vibration in [(6.24, "Moyenne :", "50", "2"),
                                            (5.68, "Écart-type :", "8,165", "0,8165")]:
        _text(ax, 4.4, y, label, size=12, ha="left")
        _text(ax, 6.38, y, temperature, size=12)
        _text(ax, 8.04, y, vibration, size=12)
    table(9.5, "Valeurs transformées", [["−1,225", "−1,225"], ["0", "0"], ["+1,225", "+1,225"]],
          ["Température", "Vibration"], GREEN)
    _arrow(ax, (3.57, 5.86), (4.02, 5.86), PURPLE)
    _arrow(ax, (8.95, 5.86), (9.38, 5.86), GREEN)
    _text(ax, 6.5, 4.36, "Pour chaque colonne : valeur transformée = (mesure − moyenne) / écart-type", size=12, weight="bold")
    _text(ax, 6.5, 3.93, "Exemple : (40 − 50) / 8,165 ≈ −1,225. Ces valeurs n’ont plus d’unité physique.", size=11, color=MUTED)
    ax.plot([.2, 12.8], [3.51, 3.51], color="#cbd5e1", linewidth=1)
    _text(ax, 6.5, 3.13, "Dans notre démonstration : X_train a la forme (512, 20, 2)", size=13, weight="bold")
    for x, width, color, label in [(.4, 3.3, BLUE, "Axe 0 : 512 séquences"),
                                  (4.02, 3.3, BLUE, "Axe 1 : 20 instants"),
                                  (7.64, 4.95, GREEN, "Axe 2 : 2 capteurs")]:
        _box(ax, x, 2.08, width, .57, color=color, face=PALE, label=label, size=11)
    _text(ax, 3.9, 1.64, "axis=(0, 1) : parcourir ces deux axes", size=11, color=BLUE)
    _text(ax, 10.11, 1.64, "Conserver une statistique par capteur", size=11, color=GREEN)
    _text(ax, 6.5, 1.07, "keepdims=True → moyenne et écart-type ont chacun la forme (1, 1, 2).", size=12, weight="bold")
    _text(ax, 6.5, .46, "Calculés sur train seulement, puis réutilisés tels quels sur validation et test.", size=12, color=PURPLE)
    _text(ax, 6.5, -.01, "On ne recalcule pas leurs statistiques : les moyennes de validation et de test ne seront pas forcément nulles.", size=11, color=MUTED)
    save(fig, "01_standardisation_axes")


def _shapes(save):
    fig, ax = _canvas("Suivre les formes : que reçoit le réseau, que renvoie-t-il ?", size=(13, 6.5), ylim=(0, 6.5))
    _text(ax, 6.5, 6.15, "On suit ici une seule séquence, du début jusqu’à la classe prédite.", color=MUTED, size=12)
    stages = [(.15, 2.65, BLUE, "Une séquence", "20 instants\n× 2 capteurs", "forme (20, 2)"),
              (3.18, 2.65, PURPLE, "SimpleRNN(16)", "Dernière mémoire\n= 16 nombres", "forme (16,)"),
              (6.21, 2.65, GREEN, "Dense(1) + sigmoid", "Un seul score\nentre 0 et 1", "forme (1,)"),
              (9.24, 3.55, ORANGE, "Seuil = 0,5", "Score < 0,5 : classe 0\nScore ≥ 0,5 : classe 1", "une décision")]
    for i, (x, width, color, title, detail, shape) in enumerate(stages):
        _box(ax, x, 2.76, width, 2.76, color=color)
        _text(ax, x + width/2, 5.03, title, size=11.5, color=color, weight="bold")
        _text(ax, x + width/2, 4.12, detail, size=11.5)
        _text(ax, x + width/2, 3.24, shape, size=12, color=color, weight="bold")
        if i < 3:
            _arrow(ax, (x+width+.04, 4.04), (stages[i+1][0]-.07, 4.04))
    _text(ax, 6.5, 2.15, "16 est la taille de la mémoire : ce n’est ni 16 instants, ni 16 capteurs.", size=13, color=PURPLE, weight="bold")
    _box(ax, .24, .40, 12.48, 1.14, color=MUTED, face=PALE)
    _text(ax, 6.5, 1.11, "Dans l’exercice : 24 instants × 3 capteurs → mémoire de 16 nombres → 1 score → 1 classe", size=11.5, weight="bold")
    _text(ax, 6.5, .64, "Démo avec plusieurs séquences à la fois : un axe de lot s’ajoute → (lot, 20, 2) → (lot, 16) → (lot, 1).", size=11, color=MUTED)
    save(fig, "01_modele_formes")


def _learning(save):
    fig, (ax, note) = plt.subplots(1, 2, figsize=(13, 5.9), gridspec_kw={"width_ratios": [1.65, 1]})
    epochs = np.arange(1, 9)
    train = np.array([.82, .60, .42, .30, .24, .20, .17, .15])
    validation = np.array([.87, .65, .48, .50, .54, .57, .60, .64])
    ax.plot(epochs, train, "o-", color=BLUE, linewidth=2, label="Entraînement : apprend")
    ax.plot(epochs, validation, "o-", color=ORANGE, linewidth=2, label="Validation : surveille")
    ax.axvspan(3.5, 8.3, color=ORANGE, alpha=.06)
    ax.scatter([3], [.48], s=160, facecolor="none", edgecolor=GREEN, linewidth=3, zorder=5)
    ax.annotate("Meilleure validation", xy=(3, .48), xytext=(1.07, .25), color=GREEN,
                fontsize=12, ha="left", va="center", arrowprops={"arrowstyle":"->", "color":GREEN, "lw":1.5})
    for count, epoch in enumerate(range(4, 9), 1):
        ax.text(epoch, .955, str(count), ha="center", va="center", color=ORANGE, fontsize=12, weight="bold")
    ax.text(6, 1.042, "Époques sans amélioration", ha="center", color=ORANGE, fontsize=11)
    ax.set(xlim=(.65, 8.3), ylim=(0, 1.12), xticks=epochs,
           xlabel="Époque — ici comptée à partir de 1", ylabel="Perte : plus bas = mieux")
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.14)
    ax.legend(loc="lower left", fontsize=11, frameon=False)
    note.set(xlim=(0, 5), ylim=(0, 6)); note.axis("off")
    _box(note, .12, 4.30, 4.76, 1.2, color=GREEN, face="#f0fdf4",
         label="Époque 3\nMeilleurs poids observés", size=12)
    _text(note, 2.5, 3.57, "Époques 4 à 8 :\n5 fois sans amélioration.", size=12)
    _box(note, .12, 1.70, 4.76, 1.2, color=ORANGE, face="#fff7ed",
         label="patience=5\nArrêt après l’époque 8", size=12)
    _text(note, 2.5, .81, "restore_best_weights=True\nOn reprend les poids de l’époque 3,\npas ceux de la dernière époque.", size=11)
    fig.suptitle("Surveiller la validation pour savoir quand s’arrêter", fontsize=16, weight="bold", y=1.01)
    fig.text(.5, .02, "Schéma illustratif, pas un résultat mesuré. Le test n’intervient pas dans le choix de l’époque.",
             ha="center", fontsize=11, color=MUTED)
    fig.subplots_adjust(wspace=.25, bottom=.18, top=.9)
    save(fig, "01_apprentissage_courbes")


def build_visual_additions(save):
    """Crée exactement sept schémas et laisse le générateur gérer les exports."""
    with plt.rc_context({"font.size": 12, "axes.titlesize": 16, "figure.facecolor": "white"}):
        for builder in (_memory, _tanh, _score, _groups, _standardisation, _shapes, _learning):
            builder(save)


if __name__ == "__main__":
    # Le mode autonome sert seulement à prévisualiser ces ajouts.
    from pathlib import Path
    import sys
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("j4_visual_preview")
    destination.mkdir(parents=True, exist_ok=True)
    def export(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=145, bbox_inches="tight")
        fig.savefig(destination / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        print(name)
    build_visual_additions(export)
