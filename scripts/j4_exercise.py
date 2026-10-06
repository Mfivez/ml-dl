"""Atelier RNN guidé : comprendre les décisions avant de compléter le code."""

from textwrap import dedent, indent


def build_exercise(md, code, image, activity, blocks):
    """Construit le même parcours pour le support vierge et son corrigé."""
    def guarded(condition, message, body):
        return (
            f"if {condition}:\n    print({message!r})\nelse:\n"
            + indent(dedent(body).strip(), "    ")
        )

    model_template = '''
model = None
forme_entree = {shape}  # TODO : forme d’une séquence, sans l’axe des exemples.
taille_memoire = {memory}  # TODO : nombre de valeurs dans le résumé du RNN.

if forme_entree is None or taille_memoire is None:
    print("Complétez la forme d’entrée et la taille du résumé.")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=forme_entree),  # Reçoit un cycle complet.
        tf.keras.layers.SimpleRNN(taille_memoire, activation="tanh"),  # Lit son ordre.
        tf.keras.layers.Dense(1, activation="sigmoid"),  # Produit un score.
    ], name="ordre_trois_capteurs")
    assert model.input_shape == (None, 24, 3)  # None : nombre d’exemples variable.
    assert model.output_shape == (None, 1)  # Un seul score par cycle entier.
    print("Entrée :", model.input_shape, "| sortie :", model.output_shape)
'''

    learning_body = dedent(blocks["TRAIN"]).strip()
    learning_template = '''
history = None
ensemble_pour_apprendre = {train}  # TODO : nom de l’ensemble qui règle les poids.
ensemble_pour_surveiller = {val}  # TODO : nom de l’ensemble qui guide l’arrêt.

if model is None:
    print("Construisez d’abord le modèle.")
elif ensemble_pour_apprendre is None or ensemble_pour_surveiller is None:
    print("Choisissez les deux ensembles : 'train', 'validation' ou 'test'.")
elif (ensemble_pour_apprendre, ensemble_pour_surveiller) != ("train", "validation"):
    print("Revoyez le rôle des ensembles avant de lancer l’apprentissage.")
else:
{training}
'''

    final_body = '''
# Les statistiques viennent toujours du train, même pour ces nouveaux cycles.
X_test_std = ((X_test - moyenne_train) / ecart_type_train).astype("float32")
scores_test = model.predict(X_test_std, verbose=0).ravel()  # Un score par cycle.
predictions_test = (scores_test >= seuil).astype("int32")  # Score → classe.
bonnes_reponses = np.sum(predictions_test == y_test)
erreurs_0_vers_1 = np.sum((y_test == 0) & (predictions_test == 1))
erreurs_1_vers_0 = np.sum((y_test == 1) & (predictions_test == 0))
print("Réponses correctes :", bonnes_reponses, "/", len(y_test))
print("Accuracy :", round(float(bonnes_reponses / len(y_test)), 3))
print("Vrais 0 annoncés 1 :", erreurs_0_vers_1)
print("Vrais 1 annoncés 0 :", erreurs_1_vers_0)
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_test, labels=[0, 1], display_labels=["classe 0", "classe 1"],
    cmap="Blues", colorbar=False,
)
plt.title("Test final — lignes : réalité ; colonnes : réponse du réseau")
plt.show()
'''
    final_template = '''
scores_test = predictions_test = None
seuil = {threshold}  # TODO : conservez la limite de décision de la démonstration.
if history is None:
    print("Terminez d’abord l’apprentissage et la comparaison sur validation.")
elif seuil is None:
    print("Choisissez le seuil avant de consulter les réponses du test.")
elif seuil != 0.5:
    print("Pour cette expérience, conservons le seuil 0.5, fixé avant le test.")
else:
{evaluation}
'''

    # Le test reste rangé : sa transformation et ses prédictions attendent le bilan.
    scaling = "\n".join(
        line for line in dedent(blocks["SCALING"]).strip().splitlines()
        if not line.strip().startswith("X_test_std =")
    )

    return [
        md('''
        # Jour 4 — Atelier : expliquer puis refaire une décision du RNN

        Nous gardons la question du matin : **la vibration vient-elle avant le
        chauffage, ou après ?** Vous allez d’abord prévoir la réponse, puis
        construire et entraîner le même réseau sur de nouveaux cycles.

        La classe `0` reste **vibration puis température** ; la classe `1`,
        **température puis vibration**. La pression est notre troisième capteur :
        elle baisse pendant le chauffage, mais **ne change pas la règle des classes**.
        La pression est une force par unité de surface ; ses valeurs artificielles
        sont ici exprimées en kilopascals (kPa), une unité de pression.

        Le code technique est fourni. Vos interventions portent sur de petites
        décisions à expliquer, pas sur la recopie d’un programme complet.
        Gardez la [démonstration](01_demo_rnn_pas_a_pas.ipynb) ouverte.
        Remplacez les `None` (« pas encore de réponse ») et les textes vides.
        Sans réponse, les cellules affichent un rappel et peuvent être exécutées.
        '''),
        image("02_parcours_exercice", "Lire, prédire, préparer, construire, apprendre, comparer, évaluer et expliquer"),
        md('''
        ## 1. Lire : que reçoit le réseau ?

        Chaque cycle possède **24 instants et 3 capteurs**. Le générateur fourni
        fabrique deux versions jumelles : mêmes mesures, ordre des événements
        inversé. Leur numéro dans `groupes` rappelle leur origine commune.

        Les outils et le chargement ci-dessous n’entraînent encore aucun réseau.
        '''),
        code(blocks["SETUP"].replace("SEED = 42", "SEED = 123")),
        code(blocks["DATA_IMPORT"]),
        code('''
        X, y, groupes = creer_cycles(
            n_paires=450, n_evenement=8, n_capteurs=3, seed=SEED,
        )  # X : mesures ; y : classes ; groupes : origine des jumeaux.
        print("Mesures :", X.shape, "| réponses :", y.shape)
        print("Cycles indépendants :", len(np.unique(groupes)))

        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for capteur, nom in enumerate(["Température (°C)", "Vibration (mm/s)", "Pression (kPa)"]):
            for indice in [0, 1]:  # Deux versions du même cycle, pas deux capteurs.
                axes[capteur].plot(X[indice, :, capteur], label=f"classe {y[indice]}")
            axes[capteur].set_ylabel(nom)
            axes[capteur].legend()
        axes[-1].set_xlabel("Instant")
        plt.tight_layout()
        plt.show()
        '''),
        activity('''
        ### À vous — lire les trois axes

        Complétez les trois nombres d’après `X.shape`. Pourquoi 900 séquences
        représentent-elles seulement 450 cycles indépendants ? Dites-le à voix haute.
        ''', '''
        nombre_sequences = nombre_instants = nombre_capteurs = None
        # TODO : un nombre par nom ; ne changez pas les données X.
        if None in (nombre_sequences, nombre_instants, nombre_capteurs):
            print("À compléter : séquences, instants, capteurs.")
        else:
            assert (nombre_sequences, nombre_instants, nombre_capteurs) == X.shape
            print("Axes correctement lus ; une classe est attendue par séquence.")
        ''', '''
        nombre_sequences, nombre_instants, nombre_capteurs = 900, 24, 3
        assert (nombre_sequences, nombre_instants, nombre_capteurs) == X.shape
        print("900 séquences : 450 cycles, chacun présenté dans deux ordres.")
        ''', "Une ligne de y répond pour un cycle entier, pas pour un instant ou un capteur."),
        md('''
        ## 2. Prédire : qu’est-ce qui doit changer ?

        Regardez les trois courbes. Quand nous échangeons les deux événements,
        nous déplaçons **tous les capteurs ensemble** : la baisse de pression
        reste avec le chauffage. Le repos du début et celui de fin ne bougent pas.
        '''),
        activity('''
        ### Avant de calculer

        Un cycle porte la classe `1`. On échange ses blocs chauffage et vibration.
        Quelle classe attend-on ? Sa moyenne par capteur change-t-elle ? Justifiez.
        ''', '''
        prediction_avant_experience = ""  # TODO : nouvelle classe et effet sur les moyennes.
        print(prediction_avant_experience or "Expliquez ce qui change et ce qui reste.")
        ''', '''
        prediction_avant_experience = (
            "Classe 0 : la vibration arrive maintenant avant le chauffage. "
            "Les moyennes restent les mêmes : seules les positions ont changé."
        )
        print(prediction_avant_experience)
        ''', "C’est la classe attendue qui change nécessairement. Un modèle peut, lui, se tromper."),
        activity('''
        ### Mémoire ou poids ?

        Nous présentons un nouveau cycle à un réseau déjà entraîné. Qu’est-ce
        qui repart de zéro ? Qu’est-ce qui est conservé ? Pourquoi ?
        ''', '''
        memoire_ou_poids = ""  # TODO : distinguez le résumé de lecture des réglages appris.
        print(memoire_ou_poids or "Pour un nouveau cycle : mémoire…, poids…")
        ''', '''
        memoire_ou_poids = (
            "La mémoire repart de zéro pour ne pas mélanger les cycles. "
            "Les poids sont conservés : ils représentent les réglages appris."
        )
        print(memoire_ou_poids)
        ''', "Pendant une prédiction, la mémoire évolue à chaque instant, mais les poids ne sont pas corrigés."),
        md('''
        ## 3. Préparer : comment garder une vérification honnête ?

        La séparation est fournie. **Train** sert à apprendre, **validation** à
        surveiller et comparer, **test** au bilan final. Les jumeaux restent ensemble :
        sinon une quasi-copie d’un exemple appris pourrait se retrouver dans le test.

        Ici, nous attendons **288 / 72 / 90 cycles**, soit **576 / 144 / 180 séquences**.
        Les effectifs diffèrent de la démonstration, pas le principe.
        '''),
        code(blocks["SPLIT"] + blocks["MASKS"]),
        activity('''
        ### Choisir et expliquer

        Option A : séparer chaque séquence au hasard. Option B : séparer les numéros
        de cycle puis récupérer leurs jumeaux. Quelle option protège le bilan ?
        ''', '''
        choix_decoupage = ""  # TODO : A ou B, avec une raison.
        print(choix_decoupage or "Choisissez une option et justifiez-la.")
        ''', '''
        choix_decoupage = "B : un cycle et son jumeau ne doivent pas traverser les ensembles."
        print(choix_decoupage)
        ''', "La séparation par origine évite la fuite de données, c’est-à-dire une information du train retrouvée dans l’évaluation."),
        code('''
        # set regroupe les numéros ; isdisjoint vérifie l’absence de numéro commun.
        assert set(cycles_train).isdisjoint(cycles_val)
        assert set(cycles_train).isdisjoint(cycles_test)
        assert set(cycles_val).isdisjoint(cycles_test)
        print("Contrôle réussi : aucun cycle ne traverse les ensembles.")
        '''),
        md('''
        Rendons maintenant les échelles comparables : **soustraire la moyenne,
        puis diviser par l’écart-type** (la dispersion). Les statistiques viennent
        du train seul. Nous réutilisons ses trois moyennes et ses trois écarts-types
        pour la validation. L’ordre et les dimensions restent identiques.

        Le dessin est un exemple numérique, pas le résultat de vos données.
        '''),
        image("01_standardisation_numerique", "Trois températures changent d’échelle, mais gardent leur ordre"),
        code(scaling),
        code('''
        assert X_train_std.shape == X_train.shape  # Aucun instant ni capteur supprimé.
        assert moyenne_train.shape == (1, 1, 3)  # Une moyenne par capteur.
        assert np.allclose(X_train_std.mean(axis=(0, 1)), 0, atol=1e-4)
        print("Contrôle réussi : mêmes dimensions et moyennes proches de zéro.")
        '''),
        md('''
        ## 4. Construire : où sont l’entrée, la mémoire et la réponse ?

        Retrouvez le schéma **Suivre les formes** de la démonstration. Ici l’entrée
        passe à `(24, 3)` ; la mémoire reste composée de **16 nombres**, la sortie
        d’un seul score. **Seize nombres ne signifient pas seize instants mémorisés.**

        `Input` décrit un cycle ; `SimpleRNN` le lit ; `Dense` transforme le résumé
        final en score. `sigmoid` ramène ce score entre 0 et 1.
        '''),
        image("01_modele_formes", "Entrée de l’exercice : 24 instants, 3 capteurs ; mémoire : 16 nombres ; sortie : un score"),
        activity("### Compléter seulement les deux tailles", model_template.format(shape="None", memory="None"),
                 model_template.format(shape="X_train_std.shape[1:]", memory="16"),
                 "shape[1:] garde les dimensions temps et capteurs, sans le nombre de séquences. Les 16 nombres résument toute la lecture."),
        md('''
        La commande `compile` ci-dessous prépare l’apprentissage : **Adam** ajuste
        les poids, la **perte** mesure les mauvaises réponses, et l’**accuracy**
        compte les bonnes classes. Elle n’entraîne pas encore le réseau.
        '''),
        code(guarded("model is None", "Complétez d’abord les deux tailles.", blocks["COMPILE"])),
        md('''
        ## 5. Apprendre : quelles données peuvent corriger les poids ?

        Choisissez les deux ensembles avant de lancer le code fourni. Il fera au
        maximum **35 passages** sur le train, par lots de 32 séquences. Il s’arrêtera
        après 5 passages sans amélioration de la perte de validation et gardera
        les meilleurs poids observés. Le test ne doit intervenir nulle part ici.
        '''),
        activity("### Choisir qui apprend et qui surveille",
                 learning_template.format(train="None", val="None", training=indent(learning_body, "    ")),
                 learning_template.format(train='"train"', val='"validation"', training=indent(learning_body, "    ")),
                 "fit corrige les poids avec le train. La validation guide l’arrêt sans fournir de corrections de poids."),
        code(guarded("history is None", "Les courbes apparaîtront après l’apprentissage.", blocks["CURVES"])),
        md('''
        Lisez vos deux courbes sans attendre un chiffre imposé. Si la perte du
        train baisse alors que celle de validation remonte durablement, le réseau
        se spécialise trop sur ses exemples : c’est le **surapprentissage**.
        '''),
        md('''
        ## 6. Comparer : le réseau réagit-il au changement d’ordre ?

        Revenons à votre prédiction de départ, sur la **validation uniquement**.
        Nous prenons la version de classe `1` de la première paire réservée :
        chauffage puis vibration. L’échange doit donc donner la classe attendue `0`.
        Le code échange les blocs `4:12` et `12:20` ; une tranche Python inclut
        sa première position et exclut la dernière. Les deux repos restent en place.

        Comparez les classes annoncées aux classes attendues. Le réseau doit-il
        rendre deux classes différentes ? Le fait-il réellement ? **Une erreur
        est un résultat à commenter, pas à cacher.**
        '''),
        code(guarded("history is None", "Entraînez d’abord le modèle.", '''
        cycle_source = X_val[1:2]  # Version de classe 1 ; la tranche garde l’axe du lot.
        cycle_permute = cycle_source.copy()  # Ne modifie jamais les données de validation.
        cycle_permute[:, 4:12, :] = cycle_source[:, 12:20, :]
        cycle_permute[:, 12:20, :] = cycle_source[:, 4:12, :]
        comparaison = np.concatenate([cycle_source, cycle_permute])
        comparaison_std = ((comparaison - moyenne_train) / ecart_type_train).astype("float32")

        poids_avant = [valeur.copy() for valeur in model.get_weights()]
        scores_comparaison = model.predict(comparaison_std, verbose=0).ravel()
        attendues = [int(y_val[1]), 1 - int(y_val[1])]  # Inverser l’ordre inverse la classe.
        for position, nom in enumerate(["Cycle original", "Événements échangés"]):
            print(nom, "| attendu :", attendues[position],
                  "| annoncé :", int(scores_comparaison[position] >= 0.5),
                  "| score :", round(float(scores_comparaison[position]), 3))

        assert all(np.array_equal(avant, apres)
                   for avant, apres in zip(poids_avant, model.get_weights()))
        print("Contrôle : predict n’a pas changé les poids.")

        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for capteur, nom in enumerate(["Température (°C)", "Vibration (mm/s)", "Pression (kPa)"]):
            for position, etiquette in enumerate(["Original", "Permuté"]):
                axes[capteur].plot(comparaison[position, :, capteur], label=etiquette)
            axes[capteur].set_ylabel(nom)
            axes[capteur].legend()
        axes[-1].set_xlabel("Instant")
        plt.tight_layout()
        plt.show()
        ''')),
        md('''
        **Un score proche de 1 ne prouve pas que la réponse est fiable.** Cette
        comparaison vérifie une réaction à l’ordre sur un exemple ; elle ne
        démontre pas que tous les cycles seront bien classés.

        Si vous souhaitez comparer un autre réglage avec le formateur, faites-le
        **maintenant sur la validation**, puis fixez votre choix avant la suite.
        '''),
        md('''
        ## 7. Évaluer : ouvrir le test une fois les choix fixés

        Conservez le seuil `0.5` de la démonstration. Le bilan calcule la part de
        réponses correctes et les deux types d’erreur. Dans la **matrice de
        confusion**, les lignes indiquent la réalité, les colonnes les réponses.
        Les cases diagonales sont les réponses justes.

        Après ce bilan, ne réglez plus le modèle selon le test : il ne serait
        alors plus une vérification indépendante de vos choix.
        '''),
        activity("### Compléter le seuil, puis lire les erreurs",
                 final_template.format(threshold="None", evaluation=indent(dedent(final_body).strip(), "    ")),
                 final_template.format(threshold="0.5", evaluation=indent(dedent(final_body).strip(), "    ")),
                 "Vrai 0 annoncé 1 signifie que le réseau a inversé l’ordre vibration puis température. Le score global ne dit pas à lui seul quelles erreurs ont été commises."),
        md('''
        ## 8. Expliquer : que vient-on réellement de démontrer ?

        Rendez les courbes d’apprentissage, la comparaison avant/après échange,
        la matrice du test et une courte conclusion. Utilisez **vos résultats**,
        même si le modèle commet des erreurs.
        '''),
        activity('''
        ### Trois phrases pour terminer

        1. Quel rôle joue la mémoire, et que gardons-nous entre deux cycles ?
        2. Que montrent votre permutation et votre bilan, sans en dire davantage ?
        3. Pourquoi ce résultat ne suffit-il pas à détecter des pannes réelles ?
        ''', '''
        conclusion = ""  # TODO : mécanisme, observation personnelle, limite.
        print(conclusion or "Appuyez vos trois phrases sur les sorties de votre expérience.")
        ''', '''
        conclusion = (
            "La mémoire résume la lecture d’un cycle ; les poids appris sont conservés entre les cycles. "
            "La permutation vérifie une réaction à l’ordre et le test mesure les erreurs sur des cycles réservés ; "
            "leurs chiffres doivent être repris dans mon compte rendu. "
            "Ces cycles artificiels et ces deux ordres ne représentent pas toute la diversité des pannes réelles."
        )
        print(conclusion)
        ''', "Même un test sans erreur ne prouve ni que le RNN est la meilleure solution, ni qu’il est prêt pour une machine réelle."),
    ]
