"""Théorie du jour 4 : un seul cycle suivi de la mesure à la décision."""


def build_theory(md, code, image, activity):
    """Retourne les cellules et activités, sans générer de fichier à l'import."""
    return [
        md('''
        # Jour 4 — Comprendre un RNN : suivre un cycle jusqu'à la décision

        Une machine peut **vibrer puis chauffer**, ou **chauffer puis vibrer**.
        Aujourd'hui, nous voulons apprendre à un programme à reconnaître cet ordre.

        Un **RNN** (*Recurrent Neural Network*, ou *réseau de neurones récurrent*) lit les mesures dans l'ordre.
        Il transporte d'un instant au suivant un petit résumé numérique : sa **mémoire**.
        Nous allons voir pourquoi cette mémoire est utile, comment elle conduit à une
        réponse, puis comment le programme apprend ses réglages.

        Nous suivrons le même cycle du début à la fin : **observer → mémoriser → répondre
        → apprendre → vérifier**. Ensuite, la [démonstration](01_demo_rnn_pas_a_pas.ipynb)
        entraînera un réseau ; l'[exercice](02_exercice_rnn_capteurs.ipynb) vous permettra
        de refaire le chemin avec un capteur supplémentaire.
        '''),
        md('''
        ## 1. Une mission précise : reconnaître l'ordre, pas prédire la suite

        Un **cycle** est ici une opération complète de la machine. Deux **capteurs**,
        des appareils de mesure, suivent sa température et sa vibration.
        La **séquence** est l'ensemble de ces mesures conservées dans leur ordre.
        Un **instant** désigne une position dans cette suite, pas forcément une seconde.
        '''),
        image("00_mission_deux_ordres", "Classe 0 : repos, vibration, température, repos. Classe 1 : repos, température, vibration, repos."),
        md('''
        Le programme reçoit **tout un cycle** et rend **une seule catégorie**, appelée
        une **classe** : `0` pour vibration puis température, `1` pour l'ordre inverse.
        Ce choix d'une catégorie s'appelle une **classification**. Les nombres `0` et
        `1` sont des étiquettes : ils n'indiquent pas la gravité d'une panne.

        La bonne réponse fournie pour apprendre est la **cible**. Nous fabriquons nos
        exemples pour comprendre le mécanisme ; ce ne sont pas des diagnostics réels.
        '''),
        activity('''
        ### À vous — quelle réponse demande-t-on ?

        Un cycle commence au repos, chauffe, vibre, puis revient au repos.
        Quelle classe attend-on ? Le programme doit-il aussi prédire la température suivante ?
        ''', '''
        reponse = ""  # TODO : indiquez la classe et ce que nous ne cherchons pas à prédire.
        print(reponse or "À compléter avec vos mots.")
        ''', '''
        reponse = "Classe 1 ; nous ne prédisons pas la prochaine température."
        print(reponse)
        ''', '''
        La température précède la vibration : c'est la classe 1. La réponse porte sur
        le cycle entier, pas sur une mesure future.
        '''),
        md('''
        ## 2. Pourquoi ne pas regarder seulement la moyenne ?

        Pour commencer, remplaçons chaque événement par un signal simple : `0` signifie
        « absent », `1` « présent ». **Ces nombres ne sont pas des degrés Celsius.**
        Chaque ligne contient `[température, vibration]`.

        | Instant | Classe 0 : vibration puis température | Classe 1 : température puis vibration |
        | --- | --- | --- |
        | 0 — repos | `[0, 0]` | `[0, 0]` |
        | 1 | `[0, 1]` | `[1, 0]` |
        | 2 | `[1, 0]` | `[0, 1]` |
        | 3 — repos | `[0, 0]` | `[0, 0]` |

        Dans les deux cycles, chaque capteur vaut une fois `1` et trois fois `0`.
        Sa **moyenne**, la somme divisée par le nombre de mesures, vaut donc `1 / 4 = 0,25`.
        La dernière ligne est aussi identique.
        '''),
        md('''
        Avec les seules moyennes ou la dernière ligne, les deux classes
        deviennent indiscernables. **L'information nécessaire a été supprimée.**

        Il faut donc lire les événements dans l'ordre et conserver quelque chose du passé.
        '''),
        md('''
        ## 3. Lire le cycle avec une petite mémoire

        Imaginons une note : « vibration déjà vue ». Quand la température arrive,
        cette note aide à reconnaître l'ordre. Un RNN ne rédige pas cette phrase :
        il garde des nombres, son **état caché**. « Caché » signifie interne au réseau,
        pas secret. Cette mémoire est un résumé, pas un enregistrement parfait.
        '''),
        image("00_cellule_rnn", "Les mesures présentes et la mémoire précédente entrent dans une règle qui produit la nouvelle mémoire."),
        md('''
        La **cellule récurrente** est cette règle de calcul, réutilisée à chaque instant.
        Son résultat revient dans le calcul suivant : c'est le sens de « récurrent ».

        Prenons une miniature avec **un seul nombre de mémoire**, noté `h`.
        Il vaut zéro avant de lire le cycle. Nous choisissons cette règle à la main :

        ```text
        nouvelle mémoire = tanh(température − vibration + 0,6 × ancienne mémoire)
        ```

        La température contribue positivement, la vibration négativement ; `0,6`
        règle la contribution du passé. Ces coefficients sont des **poids**.
        `tanh`, la tangente hyperbolique, transforme la somme en un nombre entre
        `−1` et `1` : elle garde son signe et limite son amplitude. Ce n'est pas une probabilité.

        Exécutons la règle sur **notre classe 0**. La boucle `for` lit une ligne après
        l'autre ; les deux noms récupèrent ses deux colonnes. `append` conserve un
        résultat dans la liste `trace`. `round` arrondit uniquement l'affichage.
        '''),
        code('''
        from math import tanh, exp  # Deux fonctions de calcul : mémoire puis score.

        # À chaque instant : [température, vibration], avec 0 = absent et 1 = présent.
        cycle_0 = [[0, 0], [0, 1], [1, 0], [0, 0]]
        cycle_1 = [[0, 0], [1, 0], [0, 1], [0, 0]]
        memoire = 0.0  # Nous n'avons encore rien lu de ce cycle.
        trace = []  # Gardons chaque état pour suivre son évolution.

        for temperature, vibration in cycle_0:
            memoire = tanh(temperature - vibration + 0.6 * memoire)
            trace.append(memoire)
            print([temperature, vibration], "→ mémoire :", round(memoire, 3))

        memoire_classe_0 = memoire  # Le résumé après le dernier instant.
        '''),
        md('''
        Nous lisons `0`, `−0,762`, `0,495`, puis `0,289`. Après le retour au repos,
        la mémoire n'est pas nulle : la dernière ligne est vide, mais le passé a laissé
        une trace. Ce nombre positif **n'est pas encore une classe**. Il faut lui
        associer une règle de décision.
        '''),
        activity('''
        ### À vous — même fin, passé différent

        Réutilisez exactement la même règle sur `cycle_1`, en repartant de zéro.
        Que constatez-vous sur le signe de la mémoire finale ?
        `None` signifie « pas encore de résultat » ; remplacez-le par votre calcul.
        ''', '''
        memoire_classe_1 = None  # TODO : partez de 0, puis lisez les lignes de cycle_1.
        if memoire_classe_1 is None:
            print("À compléter : lire le cycle dans l'autre ordre.")
        else:
            print("Mémoire finale :", round(memoire_classe_1, 3))
        ''', '''
        memoire_classe_1 = 0.0  # Un nouveau cycle commence avec une mémoire vide.
        for temperature, vibration in cycle_1:
            memoire_classe_1 = tanh(temperature - vibration + 0.6 * memoire_classe_1)
        print("Mémoire finale :", round(memoire_classe_1, 3))
        ''', '''
        La mémoire vaut environ −0,289, au lieu de +0,289. Cette règle distingue donc
        nos deux exemples malgré leur dernier instant identique. Cela ne prouve pas
        encore qu'elle fonctionnerait sur tous les cycles possibles.
        '''),
        md('''
        ## 4. Transformer le résumé en réponse

        Nous voulons maintenant un **score pour la classe 1**, entre `0` et `1`.
        La **sigmoïde** est une fonction qui transforme un nombre en un tel score :
        une entrée négative donne moins de `0,5`, une entrée positive plus de `0,5`.
        Ce score aide à décider ; il ne constitue pas une certitude.

        Notre règle sera `score = sigmoïde(poids_sortie × mémoire_finale)`.
        Le **seuil**, la frontière de décision, sera `0,5` : à partir de cette valeur,
        nous répondons classe `1`, sinon classe `0`.

        Reprenons notre classe 0 et sa mémoire `+0,289`. Choisissons d'abord
        `poids_sortie = +4` : ce réglage est volontairement mauvais.

        `def` nomme un calcul réutilisable et `return` renvoie son résultat.
        La fonction `exp` calcule l'exponentielle, une fonction mathématique positive
        utilisée dans la formule de la sigmoïde. Observons ici son effet sur le score.
        '''),
        code('''
        def sigmoide(nombre):
            return 1 / (1 + exp(-nombre))  # Convertit le nombre en score entre 0 et 1.

        # Même cycle et même mémoire : seul le réglage de sortie change.
        for poids_sortie in [4, -4]:
            score = sigmoide(poids_sortie * memoire_classe_0)
            classe_predite = 1 if score >= 0.5 else 0
            print("Poids :", poids_sortie, "| score :", round(score, 3),
                  "| classe prédite :", classe_predite, "| classe attendue : 0")
        '''),
        image("00_cycle_memoire_score", "Le même cycle de classe 0 produit une mémoire finale de 0,289. Avec un poids de sortie +4, le score 0,760 donne la mauvaise classe 1 ; avec −4, le score 0,240 donne la classe 0."),
        md('''
        Avec `+4`, le score vaut environ `0,760` : la réponse est fausse. Avec `−4`,
        il vaut `0,240` : la réponse est correcte. **Nous avons changé nous-mêmes un
        réglage ; aucun apprentissage automatique n'a encore eu lieu.** Le choix
        `−4` illustre une amélioration sur nos deux cycles, pas une recette universelle.

        Nous savons maintenant relier les mesures, la mémoire, le score et la classe.
        Comment trouver de bons réglages sans les choisir un par un ?
        '''),
        md('''
        ## 5. Apprendre : régler les poids, pas conserver la mémoire d'un autre cycle

        Un **modèle** est ici un programme dont certains réglages s'apprennent sur
        des exemples. Pour chaque cycle d'entraînement, il produit un score et le
        compare à la cible connue.

        La **perte** est un nombre qui mesure l'écart entre le score et la cible.
        Pour notre classe 0, un score proche de `0` est préférable à un score proche
        de `1`. L'**optimiseur** est la méthode qui utilise les erreurs pour proposer
        de petits ajustements des réglages. Le programme relit des exemples avec ces
        nouveaux réglages, sans garantie que chaque ajustement améliore chaque cycle.

        Un **biais** est un nombre ajouté à une somme avant sa transformation. Dans
        notre miniature il était nul. Le vrai réseau apprendra ses poids **et** ses
        biais : ceux qui construisent la mémoire et ceux qui produisent le score.
        '''),
        image("00_deux_boucles", "Une boucle lit les instants d'un cycle en actualisant la mémoire ; une autre utilise le score et la cible pour ajuster les réglages entre les lectures d'apprentissage."),
        activity('''
        ### À vous — qu'est-ce qui repart de zéro ?

        Nous venons de lire un cycle. Un nouveau cycle indépendant arrive.
        Que réinitialisons-nous : la mémoire ou les poids ? Et lors du test final,
        utilisons-nous les bonnes réponses pour corriger les poids ?
        ''', '''
        explication = ""  # TODO : distinguez lecture d'un cycle, apprentissage et test.
        print(explication or "À compléter avec vos mots.")
        ''', '''
        explication = (
            "La mémoire repart de zéro pour le nouveau cycle ; les poids sont conservés. "
            "Lors du test final, nous mesurons les erreurs sans corriger les poids."
        )
        print(explication)
        ''', '''
        La mémoire résume le cycle en cours. Les poids sont les règles apprises,
        réutilisables sur d'autres cycles. Réinitialiser la mémoire n'efface donc pas
        l'apprentissage. Pour un test honnête, les réponses du test ne règlent pas le modèle.
        '''),
        md('''
        ## 6. De notre miniature au réseau de la démonstration

        La démonstration gardera le même mécanisme, mais sa mémoire contiendra
        **16 nombres**, au lieu d'un seul. Chacun participe à un résumé appris.
        Cela ne veut dire ni 16 instants stockés, ni 16 capteurs : **16 est un choix
        pour cette expérience**, pas une valeur imposée par les RNN.
        '''),
        image("00_rnn_deplie", "La même cellule lit successivement les instants du cycle, transmet sa mémoire et utilise les mêmes poids jusqu'à la décision finale."),
        md('''
        Dessiner les lectures côte à côte s'appelle **déplier** le réseau : ce ne sont pas des
        réseaux différents. `SimpleRNN`, fourni par la bibliothèque **Keras**, fera
        ces calculs ; la démonstration expliquera ses commandes au moment de les utiliser.

        Il reste à ranger les mesures sans perdre leur ordre. Empilons nos deux cycles :
        '''),
        code('''
        import numpy as np  # NumPy range nos nombres dans un tableau.

        X_mini = np.array([cycle_0, cycle_1])  # Les mesures, cycle par cycle.
        y_mini = np.array([0, 1])  # Une bonne réponse pour chaque cycle.
        print("Forme des mesures :", X_mini.shape)
        print("Bonnes réponses :", y_mini)
        print("Température au troisième instant du premier cycle :", X_mini[0, 2, 0])
        '''),
        image("00_tenseur_3d", "Deux cycles, quatre instants par cycle et deux capteurs par instant : un tableau de forme (2, 4, 2)."),
        md('''
        `.shape` donne la **forme** du tableau : `(2, 4, 2)` signifie 2 cycles,
        4 instants chacun, 2 capteurs à chaque instant. Ces trois directions de
        rangement sont des **axes**. Un tableau à plusieurs axes est un **tenseur**.
        `X` nomme habituellement les mesures et `y` les cibles.

        Dans `X_mini[0, 2, 0]`, les **indices**, les positions qui commencent à zéro
        en Python, choisissent le premier cycle, le troisième instant et le premier
        capteur. La valeur obtenue est `1` : la température est présente à cet instant.
        '''),
        activity('''
        ### À vous — mesures et mémoire ne désignent pas la même chose

        Un réseau reçoit des données de forme `(80, 12, 3)` et utilise une mémoire
        de 16 nombres. Combien de cycles et de capteurs reçoit-il ? Combien de
        réponses doit-il rendre si nous classons chaque cycle entier ?
        ''', '''
        interpretation = ""  # TODO : expliquez les axes, puis le nombre de réponses.
        print(interpretation or "À compléter : distinguez entrées, mémoire et sorties.")
        ''', '''
        interpretation = (
            "80 cycles, 12 instants par cycle, 3 capteurs par instant ; "
            "80 réponses au total. Les 16 nombres sont la mémoire interne du réseau."
        )
        print(interpretation)
        ''', '''
        La forme décrit les données reçues ; la taille de mémoire décrit un choix
        interne au modèle. Nous rendons une classe par cycle, pas par instant ni par nombre de mémoire.
        '''),
        md('''
        ## 7. Vérifier sur des cycles nouveaux : la suite logique

        Notre miniature sépare deux exemples très simples. Le vrai réseau devra
        reconnaître le même ordre malgré des mesures qui varient. Nous vérifierons
        ses réponses sur des cycles qu'il n'a pas utilisés pour apprendre.

        Dans la démonstration, nous avancerons ainsi :

        1. **Observer** deux cycles et leur réponse attendue.
        2. **Réserver** des cycles : ceux pour apprendre, ceux pour surveiller
           l'apprentissage, puis ceux du test final.
        3. **Préparer** les nombres sans changer l'ordre des événements.
        4. **Entraîner** le réseau : lire, donner un score, comparer, ajuster les réglages.
        5. **Interpréter** les bonnes réponses et les erreurs sur les cycles réservés.

        Le passage aux données plus nombreuses ne change pas la question :
        **quel événement arrive en premier ?** Gardez ce fil en ouvrant
        [01 — Démonstration RNN pas à pas](01_demo_rnn_pas_a_pas.ipynb).
        '''),
    ]
