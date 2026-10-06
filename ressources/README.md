# Ressources pédagogiques

## Jeu de données `maintenance_machines.csv`

Chaque ligne représente une machine observée à un instant de référence. Les données sont synthétiques et servent uniquement à l'apprentissage.

| Colonne | Signification |
|---|---|
| `machine_id` | identifiant technique, non utilisé comme feature |
| `temperature_c` | température en degrés Celsius |
| `vibration_mm_s` | vibration en millimètres par seconde |
| `pressure_bar` | pression en bar |
| `load_pct` | charge de la machine en pourcentage |
| `age_years` | âge de la machine en années |
| `days_since_maintenance` | nombre de jours depuis la dernière maintenance |
| `failure_7d` | cible : `1` si une panne survient dans les sept jours, sinon `0` |

## Illustrations

Les illustrations PNG et SVG sont statiques afin de rester visibles avant toute exécution. Elles sont organisées par journée et intégrées directement dans les notebooks. Les figures numériques montrent les opérations, les dimensions et les erreurs à interpréter.

Le jour 4 crée des cycles synthétiques en paires : mêmes mesures et mêmes résumés, mais deux ordres différents des événements température/vibration. Chaque paire reste dans un seul ensemble train, validation ou test.
