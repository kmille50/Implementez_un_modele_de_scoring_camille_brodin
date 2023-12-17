# Projet7_OpenClassRooms

#### <i>Implémentez un modèle de scoring</i>

## Contexte du projet

La mission est de prédire le risque de faillite d'un client pour une société de crédit, "Pret à dépenser". Nous allons évaluer 5 modèles de classification binaire au moyen des métriques correspondantes (AUC, Rappel, précision etc), identifier et optimiser les hparams du meilleur modèle parmis la sélection.

En parallèle, nous créerons une API web avec un Dashboard interactif. Celui-ci devra a minima contenir les fonctionnalités suivantes :

- Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

Il nous est proposé d'utiliser un Notebook disponible sur le site de Kaggle pour le feature engineering (cf lien de fin)

## Contenu

<u>Dans ce dépôt, se trouve :</u>

- Le notebook ou code de la modélisation (du prétraitement à la prédiction), intégrant via MLFlow le tracking d’expérimentations et le stockage centralisé des modèles
- Le code générant le dashboard
- Le code permettant de déployer le modèle sous forme d'API
- Pour les applications dashboard et API, un fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers, et un fichier listant les packages utilisés seront présents dans les dossiers
- Le tableau HTML d’analyse de data drift réalisé à partir d’evidently

## Modèle de classification

Le modèle retenu pour cet exercice est le modèle GradientBoosting. Le feature engineering étant sommaire, les résultats obtenus sont perfectible. 

## Dashboard / API

J'ai utilisé deux librairies Python pour ce sujet :
- Flask
- Streamlit

## Données d'entrées

- Lien de téléchargement des données d'entrées : https://www.kaggle.com/c/home-credit-default-risk/data 
- Notebook de départ pour la partie Features Engineering : https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
