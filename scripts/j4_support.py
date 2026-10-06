"""Données artificielles et observation d’un RNN pour le jour 4.

Outils fournis : pas de dépendance à un notebook déjà exécuté.
"""

import numpy as np

def creer_cycles(n_paires=400, n_evenement=6, n_capteurs=2, seed=42):
    """Deux jumeaux par cycle : mêmes mesures, événements permutés."""
    rng = np.random.default_rng(seed)
    n_instants = 8 + 2 * n_evenement  # Repos initial/final : quatre instants chacun.
    X = np.empty((2 * n_paires, n_instants, n_capteurs), dtype="float32")
    y = np.tile([0, 1], n_paires)  # Une classe 0 et une classe 1 par paire.
    groupes = np.repeat(np.arange(n_paires), 2)  # Identifie les jumeaux.

    for paire in range(n_paires):
        cycle = rng.normal(0, 0.12, size=(n_instants, n_capteurs))
        cycle[4:4 + n_evenement, 0] += rng.uniform(2.5, 3.5)  # Température.
        cycle[4 + n_evenement:-4, 1] += rng.uniform(2.5, 3.5)  # Vibration.
        if n_capteurs == 3:
            cycle[4:4 + n_evenement, 2] -= 1.5  # La pression accompagne le chauffage.

        # Les unités brutes sont conservées pour les futurs graphiques.
        cycle[:, 0] = 45 + 2 * cycle[:, 0]
        cycle[:, 1] = 2 + 0.4 * cycle[:, 1]
        if n_capteurs == 3:
            cycle[:, 2] = 100 + cycle[:, 2]  # Pression artificielle en kPa.

        repos_debut = cycle[:4]
        chauffage = cycle[4:4 + n_evenement]
        vibration = cycle[4 + n_evenement:-4]
        repos_fin = cycle[-4:]
        X[2 * paire] = np.concatenate([repos_debut, vibration, chauffage, repos_fin])
        X[2 * paire + 1] = cycle  # Température avant vibration : classe 1.
    return X, y.astype("int32"), groupes


def suivre_memoire(model, sequence):
    """Rejouer la lecture avec les vrais poids Keras, sans les modifier.

    sequence : lot d'une séquence, forme (1, temps, capteurs).
    Renvoie les états (temps, taille_memoire) et le score final.
    La démonstration compare ce score au predict réel du modèle.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or len(sequence) != 1:
        raise ValueError("Fournir un seul cycle avec son axe de lot.")
    entree, recurrence, biais = model.layers[0].get_weights()
    sortie, biais_sortie = model.layers[1].get_weights()
    memoire = np.zeros(recurrence.shape[0], dtype=np.float32)
    trace = []
    for mesures in sequence[0]:
        # @ combine les contributions : mesures et ancienne mémoire.
        memoire = np.tanh(mesures @ entree + memoire @ recurrence + biais)
        trace.append(memoire.copy())
    valeur = float((memoire @ sortie + biais_sortie)[0])
    score = float(1 / (1 + np.exp(-np.clip(valeur, -80, 80))))
    return np.asarray(trace), score

