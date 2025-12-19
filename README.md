# Classification du cancer du sein – Projet de techniques d’apprentissage

Ce projet a été réalisé dans le cadre du cours *IFT712 - Techniques d’apprentissage*. L’objectif est de comparer plusieurs méthodes de classification supervisée sur le jeu de données **Breast Cancer Wisconsin (Diagnostic)** en respectant une démarche scientifique rigoureuse.

## Problème étudié

Nous considérons un problème de classification binaire :
prédire si une tumeur est **maligne** ou **bénigne** à partir de caractéristiques numériques extraites d’images médicales.

Jeu de données :
- Breast Cancer Wisconsin (Diagnostic)
- 569 échantillons
- 30 variables numériques
- 2 classes (benign / malignant)


## Méthodes de classification évaluées

Nous avons évalué les modèles suivants :

- Régression logistique
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Arbre de décision
- Réseau de neurones (MLP)

Chaque modèle est implémenté dans une classe dédiée (`src/models/`) et entraîné via validation croisée avec recherche d’hyperparamètres.

## Organisation du projet

Le projet est structuré de manière modulaire :

- `configs/` : fichiers YAML de configuration
- `data/` : données brutes, intermédiaires et préparées
- `src/` : code source (data, features, models, evaluation, utils)
- `scripts/` : scripts d’orchestration (prepare, train, evaluate, summary)
- `models/` :
  - `artifacts/` : modèles entraînés (non versionnés)
  - `reports/` : métriques, figures, résultats (non versionnés)
- `notebooks/` : analyse des résultats (sans ré-entraînement)
- `tests/` : tests unitaires de base

## Démarche scientifique

- Les données sont séparées en ensembles d’entraînement et de test.
- Les modèles sont entraînés uniquement sur l’ensemble d’entraînement.
- Une validation croisée stratifiée est utilisée pour la sélection
  des hyperparamètres.
- Les métriques utilisées incluent :
  - Accuracy
  - F1-macro
  - Precision-macro
  - Recall-macro
- L’évaluation finale est réalisée sur un jeu de test indépendant.

## Reproduire les résultats

### Prérequis

Avant de commencer, il est nécessaire de disposer de :

- Python ≥ 3.10
- make
- Un environnement virtuel Python

### Installation de l'environnement

Depuis la racine du projet :

```bash
python -m venv .ml_env
source .ml_env/bin/activate
pip install -e .
```

### Préparation des données

La préparation des données est automatisée.

```bash
make prepare
```

Cette commande :
- charge le jeu de données Breast Cancer Wisconsin via sklearn.datasets,
- sauvegarde les données brutes dans data/raw/,
- effectue le découpage train / test (stratifié),
- sauvegarde les fichiers dans data/interim/.

### Entraînement des modèles 

L'entraînement de tous les modèles configurés est lancé par :

```bash
make train
```

Cette étape :
- construit les pipelines prétraitement → classifieur,
- applique une validation croisée stratifiée,
- effectue une recherche d’hyperparamètres (GridSearchCV)
- sauvegarde les meilleurs modèles dans models/artifacts/
- sauvegarde les résultats de validation croisée dans models/reports/.

### Evaluation sur le jeu de test

L'évaluation finale est réalisée avec :

```bash
make evaluate
```
Cette commande :
- recharge les meilleurs modèles entraînés,
- calcule les performances sur le jeu de test,
- génère :
    - les métriques de classification (.csv et .txt),
    - les matrices de confusion,
    - les rapports de classification détaillés.

Les résultats sont stockés dans models/reports/.

### Génération du tableau comparatif global

Enfin, un tableau de synthèse est généré avec :

```bash
make summary
```

Cette étape construit un fichier summary.csv qui constitue la base de la comparaison finale des modèles. Ce fichier regroupe, pour chaque modèle :

- les métriques sur le jeu de test,
- le meilleur score obtenu en validation croisée,
- les meilleurs hyperparamètres associés.

## Résultats

### Métriques utilisées

Les modèles sont évalués à l'aide des métriques suivantes :
- Accuracy: proportion globale des prédictions correctes
- F1-score macro: moyenne non pondérée des F1-scores par classe
- Précision macro: capacité à éviter les faux positifs
- Rappel macro: capacité à détecter correctement les cas positifs
- Score de validation croisée: performance moyenne obtenue durant la recherche d'hyperparamètres.

### Tableau comparatif global

Les résultats finaux sont synthétisés dans le fichier :

```bash
models/reports/summary.csv
```

Ce tableau permet :
- une comparaison directe des modèles,
- l'identification du meilleur compromis performance/robustesse
- une lecture rapide pour le correcteur
