"""Supports progressifs RNN ; génération limitée au jour 4 et ses images."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from j4_visual_additions import build_visual_additions


ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
IMAGES = ROOT / "ressources" / "illustrations" / "jour_04"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "figure.facecolor": "white"})
BLUE, ORANGE, GREEN, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#9333ea"


def source(value):
    return textwrap.dedent(value).strip() + "\n"


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def code(value):
    return {"cell_type": "code", "metadata": {}, "source": source(value),
            "outputs": [], "execution_count": None}


def image(name, alt):
    return md(f"![{alt}](../ressources/illustrations/jour_04/{name}.png)")


def activity(prompt, starter, solution, explanation):
    return {"activity": True, "prompt": source(prompt), "starter": source(starter),
            "solution": source(solution), "explanation": source(explanation)}


def materialize(items, corrected=False):
    cells = []
    for item in items:
        if item.get("activity"):
            cells.extend([md(item["prompt"]), code(item["solution"] if corrected else item["starter"])])
            if corrected:
                cells.append(md(item["explanation"]))
        else:
            item = json.loads(json.dumps(item))
            if corrected and item["cell_type"] == "markdown":
                item["source"] = item["source"].replace("../ressources/", "../../ressources/")
                item["source"] = item["source"].replace("../scripts/", "../../scripts/")
                # Dans un corrigé, les liens restent dans le parcours corrigé.
                for name in ["00_comprendre_les_sequences_et_les_rnn", "01_demo_rnn_pas_a_pas", "02_exercice_rnn_capteurs"]:
                    item["source"] = item["source"].replace(f"({name}.ipynb)", f"({name}_corrige.ipynb)")
                if item["source"].startswith("# Jour 4"):
                    first, *rest = item["source"].splitlines()
                    item["source"] = first + " — corrigé\n" + "\n".join(rest) + "\n"
            cells.append(item)
    return cells


def write_nb(path, items, corrected=False):
    cells = materialize(items, corrected)
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(f"{path.name}:{i}:{cell['source']}".encode()).hexdigest()[:8]
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}
    if path.exists():
        # Conserver notamment le noyau et l'interpréteur déjà choisis dans VS Code.
        metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", metadata)
    nb = {"cells": cells, "metadata": metadata,
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / f"{name}.png", dpi=145, bbox_inches="tight")
    fig.savefig(IMAGES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, text, color=BLUE, width=2.5, height=1):
    x, y = xy
    ax.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor=color, linewidth=2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, start, end, color="#334155", label=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                               linewidth=1.8, color=color))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+0.15, label,
                ha="center", va="bottom", fontsize=10, color=color)


def build_images():
    # Première image : le problème se lit sans vocabulaire de réseau de neurones.
    fig, ax = plt.subplots(figsize=(12, 4))
    for y, label, stages, color in [
        (2.2, "Classe 0", ["Repos", "Vibration", "Température", "Repos"], BLUE),
        (.6, "Classe 1", ["Repos", "Température", "Vibration", "Repos"], ORANGE),
    ]:
        ax.text(-.2, y + .4, label, ha="right", va="center", weight="bold", color=color)
        for i, stage in enumerate(stages):
            x = .2 + i * 2.5
            box(ax, (x, y), stage, color, 2.1, .8)
            if i < 3:
                arrow(ax, (x + 2.1, y + .4), (x + 2.5, y + .4))
    ax.text(5, -.05, "Même départ, mêmes événements, même fin : seule leur place change.", ha="center")
    ax.set(xlim=(-1.6, 10.5), ylim=(-.4, 3.5)); ax.axis("off")
    ax.set_title("Notre question : lequel des deux événements arrive en premier ?")
    save(fig, "00_mission_deux_ordres")

    fig, ax = plt.subplots(figsize=(12, 3.3))
    stages = ["1. Observer\nles deux ordres", "2. Comprendre\nla mémoire", "3. Apprendre\nsur des exemples", "4. Vérifier\nsur d’autres cycles"]
    for i, stage in enumerate(stages):
        x = i * 3
        box(ax, (x, 1), stage, [BLUE, PURPLE, ORANGE, GREEN][i], 2.55, 1.1)
        if i < 3:
            arrow(ax, (x + 2.55, 1.55), (x + 3, 1.55))
    ax.text(5.7, .35, "Comprendre ensemble → regarder la démonstration → refaire avec un troisième capteur", ha="center")
    ax.set(xlim=(-.2, 11.8), ylim=(0, 2.7)); ax.axis("off")
    ax.set_title("Le fil de la journée : chaque étape répond à la question précédente")
    save(fig, "00_fil_journee")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for x, label, color in [(0, "Pendant la lecture d’un cycle", PURPLE), (6, "Pendant l’apprentissage", ORANGE)]:
        ax.text(x + 2.6, 3.6, label, ha="center", weight="bold", color=color)
    box(ax, (.2, 2), "Mesure + mémoire\n→ nouvelle mémoire", PURPLE, 5, 1)
    ax.text(2.7, 1.3, "La mémoire change à chaque instant.\nLes poids restent les mêmes pendant cette lecture.", ha="center", va="center")
    ax.text(2.7, .3, "Au début d’un autre cycle : mémoire à zéro.", ha="center", fontsize=10)
    box(ax, (6.2, 2), "Score + bonne réponse\n→ erreur → correction des poids", ORANGE, 5, 1)
    ax.text(8.7, 1.3, "Les poids se règlent sur les exemples\nd’entraînement et servent aux cycles suivants.", ha="center", va="center")
    ax.text(8.7, .3, "Lors du test : plus de correction des poids.", ha="center", fontsize=10)
    ax.set(xlim=(-.1, 11.5), ylim=(-.1, 4)); ax.axis("off")
    ax.set_title("Ne pas confondre : la mémoire de lecture et les réglages appris", pad=18)
    save(fig, "00_memoire_et_apprentissage")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, values, title, color in zip(axes, [[10,20,30,40,50],[50,40,30,20,10]],
                                      ["La température monte", "La température descend"], [ORANGE, BLUE]):
        ax.plot(range(1,6), values, "o-", color=color)
        ax.axhline(30, linestyle="--", color=GREEN, label="Moyenne = 30")
        for x,y in enumerate(values,1): ax.annotate(str(y),(x,y),xytext=(0,8),textcoords="offset points",ha="center")
        ax.set(title=title, xlabel="Instant", xticks=range(1,6), ylim=(0,65))
        ax.legend(loc="upper center")
    axes[0].set_ylabel("Température")
    fig.suptitle("Mêmes cinq valeurs et même moyenne ; l’ordre change le sens")
    fig.tight_layout()
    save(fig,"00_ordre_compte")

    fig,ax=plt.subplots(figsize=(10,3.5))
    values=[12,13,15,18,22,27,31]
    for row,start,color in [(0,0,BLUE),(1,1,GREEN),(2,2,ORANGE)]:
        for i,value in enumerate(values):
            ax.add_patch(Rectangle((i,row),.9,.7,facecolor=color if start<=i<start+4 else "#f1f5f9",alpha=.25))
            ax.text(i+.45,row+.35,str(value),ha="center",va="center")
        ax.add_patch(Rectangle((start-.04,row-.04),3.98,.78,fill=False,edgecolor=color,linewidth=2))
        ax.text(-.2,row+.35,f"Fenêtre {row+1}",ha="right",va="center")
    ax.set(xlim=(-1.5,7.2),ylim=(-.3,3.4))
    ax.invert_yaxis();ax.axis("off")
    ax.set_title("Longueur = 4 ; décalage = 1 : les cases sélectionnées se déplacent")
    save(fig,"00_fenetre_temporelle")

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    matrices=[[[40,1.2],[41,1.3],[43,1.5]],[[38,1.1],[38,1.1],[39,1.0]]]
    for i,(ax,values) in enumerate(zip(axes,matrices)):
        ax.axis("off")
        table=ax.table(cellText=values,colLabels=["température", "vibration"],
                       rowLabels=["instant 0","instant 1","instant 2"],loc="center",cellLoc="center")
        table.scale(1,2)
        ax.set_title(f"X[{i}] : une séquence de forme (3, 2)")
    fig.suptitle("X.shape = (2, 3, 2) : 2 tableaux × 3 lignes temporelles × 2 colonnes")
    fig.text(.5,.08,"X[0, 2, 0] = 43 → séquence 0, instant 2, variable 0",ha="center",color=ORANGE)
    fig.subplots_adjust(wspace=.65, top=.78, bottom=.2)
    save(fig,"00_tenseur_3d")

    fig,ax=plt.subplots(figsize=(10,4))
    box(ax,(.2,2.2),"Mesures\nde cet instant",BLUE,2.5,.9)
    box(ax,(.2,.2),"Résumé\ndu passé",PURPLE,2.5,.9)
    box(ax,(4,1.2),"Même règle\nde calcul",ORANGE,2.7,1.2)
    box(ax,(8,1.3),"Résumé\nmis à jour",GREEN,2.6,1)
    arrow(ax,(2.7,2.65),(4,2.05),BLUE)
    arrow(ax,(2.7,.65),(4,1.55),PURPLE)
    arrow(ax,(6.7,1.8),(8,1.8),GREEN)
    ax.text(9.3,.8,"vers l’instant suivant",ha="center",fontsize=10)
    ax.set(xlim=(0,11),ylim=(0,3.6));ax.axis("off")
    ax.set_title("Deux informations entrent ensemble dans une seule cellule")
    save(fig,"00_cellule_rnn")

    fig,ax=plt.subplots(figsize=(11,4))
    for i in range(3):
        x=.6+i*3.2
        box(ax,(x,1.3),f"Même cellule\ninstant {i+1}",ORANGE,2.2,1)
        arrow(ax,(x+1.1,.2),(x+1.1,1.3),BLUE)
        ax.text(x+1.1,.1,f"mesure x{i+1}",ha="center",va="top")
        if i<2:arrow(ax,(x+2.2,1.8),(x+3.2,1.8),PURPLE,f"h{i+1}")
    arrow(ax,(8.1,2.3),(8.1,3.3),GREEN)
    ax.text(8.1,3.45,"Dernier état → décision",ha="center")
    ax.text(3.7,2.9,"Les mêmes poids sont réutilisés",ha="center",color=ORANGE)
    ax.set(xlim=(0,10),ylim=(-.4,4));ax.axis("off")
    ax.set_title("Déplier le RNN permet de suivre la mémoire dans le temps")
    save(fig,"00_rnn_deplie")

    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    updates=np.arange(0,11)
    for ax,mult,title,color in zip(axes,[.5,1.5],["Le signal s’affaiblit", "Le signal s’amplifie"],[BLUE,ORANGE]):
        values=mult**updates
        ax.plot(updates,values,"o-",color=color)
        ax.set(title=title,xlabel="Nombre de pas remontés",ylabel="Taille d’un signal illustratif")
        ax.grid(alpha=.2)
    fig.suptitle("Illustration simplifiée : multiplier plusieurs fois peut éteindre ou amplifier")
    fig.tight_layout()
    save(fig,"00_gradient_temps")

    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,gate,title in zip(axes,[.1,.9],["Porte presque fermée", "Porte presque ouverte"]):
        ax.bar(["signal reçu","signal transmis"],[1,gate],color=[PURPLE,GREEN])
        ax.set(ylim=(0,1.2),ylabel="Quantité illustrative",title=f"{title} : {gate}")
        ax.text(1,gate+.05,f"1 × {gate} = {gate}",ha="center")
    fig.suptitle("Une porte règle la part d’information transmise (idée utilisée par GRU et LSTM)")
    fig.tight_layout();save(fig,"00_choisir_cellule")

    fig,ax=plt.subplots(figsize=(11,3))
    stages=["Observer\nX et y", "Séparer\nles cycles", "Standardiser\ntrain seul", "Apprendre\nSimpleRNN", "Évaluer\ntest"]
    for i,stage in enumerate(stages):
        box(ax,(i*2.2,.8),stage,[BLUE,PURPLE,GREEN,ORANGE,BLUE][i],1.9,1.1)
        if i<4:arrow(ax,(i*2.2+1.9,1.35),(i*2.2+2.2,1.35))
    ax.set(xlim=(-.2,11),ylim=(0,2.6));ax.axis("off")
    ax.set_title("Un seul fil : données → expérience honnête → décision")
    save(fig,"01_pipeline_demo")
    # L’exercice reprend le même protocole ; ce visuel ne change pas le mécanisme.
    fig,ax=plt.subplots(figsize=(10,3))
    ax.axis("off")
    vals=[["1", "Lire la forme", "(séquences, temps, capteurs)"],
          ["2", "Séparer les cycles", "pas de jumeaux dans deux ensembles"],
          ["3", "Préparer", "moyenne et écart-type du train"],
          ["4", "Construire / entraîner", "surveiller la validation"],
          ["5", "Évaluer / interpréter", "test final et erreurs concrètes"]]
    table=ax.table(cellText=vals,colLabels=["Étape","Action","Contrôle"],loc="center",cellLoc="left",colWidths=[.08,.32,.6])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.75)
    save(fig,"02_parcours_exercice")

    fig,axes=plt.subplots(2,2,figsize=(11,5),sharex=True)
    t=np.arange(20);base=np.zeros((20,2))
    base[4:10,0]=1;base[10:16,1]=1
    reversed_order=np.concatenate([base[:4],base[10:16],base[4:10],base[16:]])
    for col,sequence in enumerate([base,reversed_order]):
        for row,(name,color) in enumerate([("température",ORANGE),("vibration",BLUE)]):
            axes[row,col].plot(t,sequence[:,row],"o-",color=color)
            axes[row,col].set(ylabel=name,ylim=(-.15,1.4),xticks=[0,4,10,16,19])
            axes[row,col].axvspan(4,9,alpha=.08,color=GREEN)
            axes[row,col].axvspan(10,15,alpha=.08,color=PURPLE)
        axes[0,col].set_title(["Classe 1 : température puis vibration", "Classe 0 : vibration puis température"][col])
        axes[1,col].set_xlabel("Instant")
    fig.suptitle("Schéma simplifié : même moyenne et mêmes valeurs finales, ordre différent")
    fig.tight_layout();save(fig,"01_ordre_evenements")

    fig,ax=plt.subplots(figsize=(9,3.5))
    values=[40,50,60];scaled=[-1.225,0,1.225]
    ax.axis("off")
    table=ax.table(cellText=[[v,50,8.165,round(s,3)] for v,s in zip(values,scaled)],
        colLabels=["Température","Moyenne","Écart-type","(valeur − moyenne) / écart-type"],loc="center",cellLoc="center",colWidths=[.16,.15,.18,.51])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
    ax.set_title("Standardiser : les écarts changent d’unité, l’ordre reste le même")
    save(fig,"01_standardisation_numerique")

    fig,ax=plt.subplots(figsize=(6,4.4))
    matrix=np.array([[7,2],[1,10]])
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=12)
    descriptions=[["7 classes 0\nbien reconnues","2 classes 0\nprises pour 1"],["1 classe 1\nprise pour 0","10 classes 1\nbien reconnues"]]
    for i in range(2):
        for j in range(2):ax.text(j,i,descriptions[i][j],ha="center",va="center",fontsize=11)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["prédit 0","prédit 1"],yticklabels=["vrai 0","vrai 1"],title="Exemple : lire une matrice de confusion")
    save(fig,"02_matrice_confusion_numerique")


SETUP = '''
import os  # Réglages avant de charger les outils de calcul.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les messages techniques.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Utilise le processeur du poste.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Même réglage de calcul entre essais.
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"  # Graphiques visibles.
import numpy as np  # Tableaux de nombres et calculs sur leurs axes.
import pandas as pd  # Tableau lisible pour l’historique d’apprentissage.
import matplotlib.pyplot as plt  # Graphiques des séquences et des erreurs.
import tensorflow as tf  # Calcul et apprentissage des réseaux de neurones.
from IPython.display import display  # Affiche les tableaux dans le notebook.

from sklearn.model_selection import train_test_split  # Sépare les cycles.
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

SEED = 42
tf.keras.utils.set_random_seed(SEED)  # Fixe Python, NumPy et TensorFlow.
print("TensorFlow :", tf.__version__)
'''




SPLIT = '''
cycles_uniques = np.unique(groupes)  # Une seule entrée par paire de jumeaux.
cycles_train_val, cycles_test = train_test_split(
    cycles_uniques, test_size=0.20, random_state=SEED
)
cycles_train, cycles_val = train_test_split(
    cycles_train_val, test_size=0.20, random_state=SEED
)
'''


MASKS = '''
# np.isin indique pour chaque séquence si son cycle appartient à l’ensemble voulu.
masque_train = np.isin(groupes, cycles_train)
masque_val = np.isin(groupes, cycles_val)
masque_test = np.isin(groupes, cycles_test)

X_train, y_train = X[masque_train], y[masque_train]
X_val, y_val = X[masque_val], y[masque_val]
X_test, y_test = X[masque_test], y[masque_test]
print("Train / validation / test :", X_train.shape, X_val.shape, X_test.shape)
'''


SCALING = '''
# Les axes 0 et 1 parcourent les séquences et le temps, sans mélanger les capteurs.
moyenne_train = X_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
ecart_type_train = X_train.std(axis=(0, 1), keepdims=True, dtype=np.float64)

# float64 stabilise les statistiques ; float32 suffit ensuite pour le réseau.
X_train_std = ((X_train - moyenne_train) / ecart_type_train).astype("float32")
X_val_std = ((X_val - moyenne_train) / ecart_type_train).astype("float32")
X_test_std = ((X_test - moyenne_train) / ecart_type_train).astype("float32")
print("Statistiques par capteur :", moyenne_train.shape)
print("Moyennes après transformation :", X_train_std.mean(axis=(0, 1)).round(3))
'''


MODEL = '''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train_std.shape[1:]),  # Temps et capteurs.
    tf.keras.layers.SimpleRNN(16, activation="tanh"),  # Mémoire de 16 nombres.
    tf.keras.layers.Dense(1, activation="sigmoid"),  # Score de la classe 1.
], name="ordre_deux_evenements")
'''


COMPILE = '''
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),  # Petites corrections.
    loss="binary_crossentropy",  # Pénalise une mauvaise réponse binaire.
    metrics=["accuracy"],  # Proportion de classes correctement reconnues.
)
'''


TRAIN = '''
arret = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    X_train_std, y_train,  # Seul cet ensemble modifie les poids.
    validation_data=(X_val_std, y_val),  # La validation mesure, sans apprendre.
    epochs=35, batch_size=32, callbacks=[arret], verbose=0,
)
print("Époques exécutées :", len(history.history["loss"]))
'''


CURVES = '''
historique = pd.DataFrame(history.history)  # Une ligne par époque.
axe = historique[["loss", "val_loss"]].plot(figsize=(8, 3.5), marker="o")
axe.set(xlabel="Époque (indexée à partir de 0)", ylabel="Perte", title="Train et validation")
axe.grid(alpha=0.2)
plt.show()
'''


EVAL = '''
probabilites_test = model.predict(X_test_std, verbose=0).ravel()  # (n, 1) → (n,).
predictions_test = (probabilites_test >= 0.5).astype("int32")  # Score → classe.
print(classification_report(
    y_test, predictions_test,
    labels=[0, 1], target_names=["vibration puis température", "température puis vibration"],
    digits=3, zero_division=0,
))
affichage = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_test, labels=[0, 1], display_labels=["classe 0", "classe 1"],
    cmap="Blues", colorbar=False,
)
plt.title("Résultat final sur les cycles de test")
plt.show()
'''


BASELINES = '''
from sklearn.pipeline import make_pipeline  # Enchaîne transformation et classifieur.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vues = {
    "Dernier instant": (X_train[:, -1, :], X_val[:, -1, :]),
    # float64 évite de confondre l’ordre avec des écarts d’arrondi de la somme.
    "Moyenne temporelle": (X_train.mean(axis=1, dtype=np.float64),
                           X_val.mean(axis=1, dtype=np.float64)),
}
for nom, (entree_train, entree_val) in vues.items():
    reference = make_pipeline(StandardScaler(), LogisticRegression())
    reference.fit(entree_train, y_train)
    score = accuracy_score(y_val, reference.predict(entree_val))
    print(f"{nom} : accuracy validation = {score:.0%}")
'''


DATA_IMPORT = '''
from pathlib import Path  # Décrit les dossiers sur notre ordinateur.
import sys  # Indique à Python où trouver les outils fournis.

# Cherche le dossier du cours, depuis un support ou depuis un corrigé.
ROOT = next((d for d in (Path.cwd(), *Path.cwd().parents)
             if (d / "scripts" / "j4_support.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Ouvrez le dossier complet du cours : scripts/j4_support.py doit être présent.")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from j4_support import creer_cycles, suivre_memoire  # Outils fournis, expliqués dans le cours.
'''


def main():
    from j4_theory import build_theory
    from j4_demo import build_demo
    from j4_exercise import build_exercise
    from j4_progression_visuals import build_progression_visuals

    build_images()
    build_visual_additions(save)
    build_progression_visuals(save)
    blocks = {name: globals()[name] for name in (
        "SETUP", "DATA_IMPORT", "SPLIT", "MASKS", "SCALING", "MODEL",
        "COMPILE", "TRAIN", "CURVES", "EVAL", "BASELINES")}
    supports = [
        ("00_comprendre_les_sequences_et_les_rnn", build_theory(md, code, image, activity)),
        ("01_demo_rnn_pas_a_pas", build_demo(md, code, image, activity, blocks)),
        ("02_exercice_rnn_capteurs", build_exercise(md, code, image, activity, blocks)),
    ]
    for name, items in supports:
        write_nb(ROOT / "jour_04" / f"{name}.ipynb", items)
        write_nb(ROOT / "corrections" / "jour_04" / f"{name}_corrige.ipynb", items, corrected=True)
    print("Jour 4 : trois supports et trois corrigés synchronisés, illustrations PNG/SVG.")


if __name__ == "__main__":
    main()
