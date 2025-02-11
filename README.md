# Détection et Analyse des Biais dans un Moteur de Recommandations

## Description
Ce projet vise à analyser et corriger les biais dans un moteur de recommandations, un enjeu crucial dans le domaine de l'intelligence artificielle. Nous avons développé une application interactive qui permet de visualiser les biais présents dans un système de recommandations et d’apporter des solutions pour les réduire.

Le projet exploite des compétences en machine learning, bases de données et visualisation. L'objectif final est de créer une interface utilisateur intuitive pour présenter les recommandations et les analyses de biais.

## Fonctionnalités Principales
- **Moteur de recommandations :** Implémentation de recommandations basées sur le filtrage collaboratif et le contenu.
- **Analyse des biais :** Identification des biais liés aux données ou aux utilisateurs.
- **Visualisation interactive :** Tableau de bord pour explorer les recommandations et les biais.

## Technologies Utilisées
- **Langages :** Python (Pandas, NumPy, Scikit-learn, Matplotlib)
- **Base de données :** SQLite
- **Frameworks :** Dash
- **Dataset :** MovieLens

## Structure du Projet
1. **Préparation et Collecte des Données :**
   - Extraction des données depuis MovieLens.
   - Nettoyage et structuration avec Pandas.
   - Stockage dans une base de données relationnelle.

2. **Implémentation du Moteur de Recommandations :**
   - Filtrage collaboratif : Algorithmes basés sur les similarités entre utilisateurs ou items.
   - Recommandations basées sur le contenu : Utilisation des caractéristiques des items.

3. **Analyse des Biais :**
   - Identification des biais dans les données (ex : sur-représentation de certains groupes).
   - Analyse des biais utilisateurs (ex : préférences culturelles).
   - Proposition de solutions pour réduire ces biais.

4. **Visualisation et Interface Utilisateur :**
   - Création d'un tableau de bord interactif pour présenter les résultats.
   - Graphiques et analyses à partir des biais identifiés.

5. **Documentation et Mise en Valeur :**
   - Documentation professionnelle sous forme de README.
   - Publication sur LinkedIn et ajout au portfolio.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone <URL-du-dépôt>
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Configurez la base de données :
   - Créez une base de données PostgreSQL ou SQLite.
   - Importez les données nettoyées.
4. Lancez l’application :
   ```bash
   python ./dashboard/dash_interface.py
   ```

## Utilisation
- **Recommandations :** Consultez les recommandations personnalisées.
- **Analyse des biais :** Visualisez les biais identifiés et leurs impacts.
- **Exploration :** Utilisez les filtres pour explorer les données et analyser les solutions proposées.

## Résultats Attendues
- Réduction des biais présents dans les recommandations.
- Compréhension approfondie des impacts des biais sur les utilisateurs.
- Amélioration de la diversité et de l’équité dans le moteur de recommandations.

## Contribution
Les contributions sont les bienvenues ! Merci de soumettre vos suggestions via une pull request ou d’ouvrir une issue pour discuter des changements proposés.

## Équipe
- **Nicolas SILINOU** (Responsable de projet, Développeur principal)

## Licence
Ce projet est sous licence MIT. Veuillez consulter le fichier LICENSE pour plus d’informations.

