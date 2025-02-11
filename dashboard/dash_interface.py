import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency
from math import sqrt
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import os

# Chargement des données
# Remplacez par vos fichiers ou sources réelles
data_path = os.path.join(os.getcwd(), "data/")
ratings = pd.read_csv(data_path + "cleaned_ratings.csv")
movies = pd.read_csv(data_path + "cleaned_movies.csv")
users = pd.read_csv(data_path + "cleaned_users.csv")

# Préparer le corpus textuel pour l'approche basée sur le contenu
genre_columns = [col for col in movies.columns if col not in ["movie_id", "movie_title", "release_date", "imdb_url", "release_year"]]
def genres_to_keywords(row):
    keywords = []
    for genre in genre_columns:
        if row[genre] == 1:
            keywords.append(genre)
    return " ".join(keywords)

movies["genres"] = movies.apply(genres_to_keywords, axis=1)
movies["corpus"] = movies["movie_title"] + " " + movies["genres"]

# Calcul des similarités cosinus pour l'approche basée sur le contenu
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.8, min_df=0.02)
movie_features = vectorizer.fit_transform(movies["corpus"])
movie_similarity = cosine_similarity(movie_features)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies["movie_id"], columns=movies["movie_id"])

# Création de la matrice utilisateur-film pour l'approche collaborative
user_movie_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Création de la matrice item-film pour le filtrage basé sur les items
item_movie_matrix = user_movie_matrix.T
item_similarity = cosine_similarity(item_movie_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=item_movie_matrix.index, columns=item_movie_matrix.index)

# Fonction pour recommander des films (basé sur le contenu)
def recommend_similar_movies(movie_id, movie_similarity_df, movies, k=5):
    if movie_id not in movie_similarity_df.index:
        return []
    movie_similarities = movie_similarity_df.loc[movie_id]
    similar_movies = movie_similarities.sort_values(ascending=False)
    similar_movies = similar_movies[similar_movies.index != movie_id]  # Exclure le film lui-même
    recommended_movies = movies[movies["movie_id"].isin(similar_movies.head(k).index)]["movie_title"].tolist()
    return recommended_movies

# Fonction pour prédire une évaluation (user-based)
def predict_user_rating(user_id, movie_id, user_movie_matrix, user_similarity_df, k=5):
    if movie_id not in user_movie_matrix.columns:
        return None
    user_similarities = user_similarity_df.loc[user_id]
    similar_users = user_similarities[user_movie_matrix[movie_id].notna()].nlargest(k+1).iloc[1:]
    similar_users_ratings = user_movie_matrix.loc[similar_users.index, movie_id]
    weighted_sum = (similar_users_ratings * similar_users).sum()
    similarity_sum = similar_users.sum()
    return weighted_sum / similarity_sum if similarity_sum != 0 else None

# Fonction pour prédire une évaluation (item-based)
def predict_item_rating(user_id, movie_id, user_movie_matrix, item_similarity_df, k=5):
    if movie_id not in user_movie_matrix.columns:
        return None
    user_ratings = user_movie_matrix.loc[user_id]
    movie_similarities = item_similarity_df[movie_id]
    rated_movies = user_ratings[user_ratings.notna()].index
    similar_movies = movie_similarities[rated_movies].nlargest(k)
    weighted_sum = (user_ratings[similar_movies.index] * similar_movies).sum()
    similarity_sum = similar_movies.sum()
    return weighted_sum / similarity_sum if similarity_sum != 0 else None

# Fonction pour recommander des films (collaboratif - user-based)
def recommend_collaborative_user_movies(user_id, user_movie_matrix, user_similarity_df, movies, k=5):
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = {}
    for movie_id in unrated_movies:
        predicted_rating = predict_user_rating(user_id, movie_id, user_movie_matrix, user_similarity_df, k)
        if predicted_rating is not None:
            recommendations[movie_id] = predicted_rating
    top_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
    recommended_movies = movies[movies["movie_id"].isin([movie[0] for movie in top_movies])]["movie_title"].tolist()
    return recommended_movies

# Fonction pour recommander des films (collaboratif - item-based)
def recommend_collaborative_item_movies(user_id, user_movie_matrix, item_similarity_df, movies, k=5):
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = {}
    for movie_id in unrated_movies:
        predicted_rating = predict_item_rating(user_id, movie_id, user_movie_matrix, item_similarity_df, k)
        if predicted_rating is not None:
            recommendations[movie_id] = predicted_rating
    top_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
    recommended_movies = movies[movies["movie_id"].isin([movie[0] for movie in top_movies])]["movie_title"].tolist()
    return recommended_movies

# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Analyse des Biais et Recommandations"

# Définition des composants de navigation
navbar = dbc.NavbarSimple(
    brand="Moteur de Recommandations",
    brand_href="/",
    color="primary",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Accueil", href="/")),
        dbc.NavItem(dbc.NavLink("Recommandations", href="/recommandations")),
        dbc.NavItem(dbc.NavLink("Analyse des biais", href="/analyse-biais")),
    ],
)

# Layout de l'application
app.layout = dbc.Container([
    dcc.Location(id="url"),  # Composant pour la gestion de l'URL
    navbar,
    html.Div(id="page-content")  # Contenu de la page actuelle
], fluid=True)

# Contenus des pages
home_page = html.Div([
    html.H1("Bienvenue dans l'application de Recommandations !", className="mt-4"),
    html.P("Explorez les recommandations et analysez les biais dans le moteur.", className="lead"),
])

recommandations_page = html.Div([
    html.H2("Recommandations"),
    html.Label("Choisissez une approche :"),
    dcc.Dropdown(
        id="approach-selector",
        options=[
            {"label": "Basée sur le contenu", "value": "content"},
            {"label": "Collaboratif (User-Based)", "value": "collaborative_user"},
            {"label": "Collaboratif (Item-Based)", "value": "collaborative_item"}
        ],
        value="content",  # Approche par défaut
        className="mt-2"
    ),
    html.Label("Recherchez un film ou un utilisateur :"),
    dcc.Dropdown(
        id="input-selector",
        options=[{"label": title, "value": movie_id} for movie_id, title in zip(movies["movie_id"], movies["movie_title"])],
        placeholder="Tapez pour rechercher un film",
        searchable=True  # Permet une recherche dynamique
    ),
    html.Div(id="recommandations-output", className="mt-4"),
])

analyse_biais_page = html.Div([
    html.H2("Analyse des Biais"),
    html.Div([
        html.H4("Introduction"),
        html.P("Les systèmes de recommandation sont devenus omniprésents dans divers domaines, notamment le commerce en ligne, le streaming multimédia et les réseaux sociaux. Cependant, ces systèmes peuvent introduire des biais qui affectent l'équité et la diversité des recommandations. Ce rapport vise à identifier, analyser et corriger ces biais en se concentrant sur trois principaux aspects :"),
        html.Ul([
            html.Li("Genre des utilisateurs : Les biais liés au genre peuvent conduire à des recommandations déséquilibrées entre hommes et femmes."),
            html.Li("Groupes d'âge : Certains groupes d'âge peuvent être favorisés ou marginalisés."),
            html.Li("Popularité des films : Les films les plus populaires peuvent être sur-représentés, réduisant la diversité des recommandations."),
        ]),
        html.P("Objectif : Identifier et corriger les biais potentiels dans un moteur de recommandations."),
    ], className="mb-4"),
    html.Div([
        html.H4("Distribution des évaluations par film"),
        dcc.Graph(id="distribution-evaluations-film")
    ]),
    html.Div([
        html.H4("Distribution des évaluations par utilisateur"),
        dcc.Graph(id="distribution-evaluations-user")
    ]),
    html.Div([
        html.H4("Répartition des films par genre"),
        dcc.Graph(id="genre-distribution")
    ]),
    html.Div([
        html.H4("Analyse démographique : Notes moyennes par genre"),
        dcc.Graph(id="gender-rating-mean")
    ]),
    html.Div([
        html.H4("Analyse démographique : Distribution des notes par genre"),
        dcc.Graph(id="gender-rating-distribution")
    ]),
    html.Div([
        html.H4("Analyse démographique : Notes moyennes par groupe d'âge"),
        dcc.Graph(id="age-group-rating-mean")
    ]),
    html.Div([
        html.H4("Analyse démographique : Distribution des notes par groupe d'âge"),
        dcc.Graph(id="age-group-rating-distribution")
    ]),
    html.Div([
        html.H4("Analyse statistique : Résultats des tests de Chi²"),
        html.Div(id="chi2-results")
    ]),
    html.Div([
        html.H4("Tableau de contingence : Genre vs. Évaluations"),
        dcc.Graph(id="heatmap-gender")
    ]),
    html.Div([
        html.H4("Tableau de contingence : Âge vs. Évaluations"),
        dcc.Graph(id="heatmap-age")
    ]),
])


# Callbacks
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/recommandations":
        return recommandations_page
    elif pathname == "/analyse-biais":
        return analyse_biais_page
    else:
        return home_page

@app.callback(
    Output("input-selector", "options"),
    [Input("approach-selector", "value")]
)
def update_input_selector_options(selected_approach):
    if selected_approach == "content":
        # Charger les films pour l'approche basée sur le contenu
        return [{"label": title, "value": movie_id} for movie_id, title in zip(movies["movie_id"], movies["movie_title"])]
    elif selected_approach in ["collaborative_user", "collaborative_item"]:
        # Charger les utilisateurs pour les approches collaboratives
        return [{"label": f"Utilisateur {user_id}", "value": user_id} for user_id in users["user_id"]]
    else:
        return []

@app.callback(
    Output("recommandations-output", "children"),
    [Input("approach-selector", "value"), Input("input-selector", "value")]
)
def update_recommendations(selected_approach, selected_input):
    if selected_input is None:
        return "Veuillez sélectionner un utilisateur ou un film."

    if selected_approach == "content":
        recommended_movies = recommend_similar_movies(selected_input, movie_similarity_df, movies)
    elif selected_approach == "collaborative_user":
        recommended_movies = recommend_collaborative_user_movies(selected_input, user_movie_matrix, user_similarity_df, movies)
    elif selected_approach == "collaborative_item":
        recommended_movies = recommend_collaborative_item_movies(selected_input, user_movie_matrix, item_similarity_df, movies)
    else:
        return "Approche non valide sélectionnée."

    return html.Ul([html.Li(movie) for movie in recommended_movies])

@app.callback(
    Output("distribution-evaluations-film", "figure"),
    [Input("url", "pathname")]
)
def update_distribution_evaluations(pathname):
    if pathname == "/analyse-biais":
        ratings_per_movie = ratings['item_id'].value_counts()
        fig = px.histogram(
            ratings_per_movie, nbins=10, title='Distribution des évaluations par film',
            labels={'value': "Nombre d'évaluations par film", 'count': "Nombre de films"},
            log_y=False,
            text_auto=True
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(xaxis_title="Nombre d'évaluations", yaxis_title="Nombre de films")
        return fig
    return {}

@app.callback(
    Output("distribution-evaluations-user", "figure"),
    [Input("url", "pathname")]
)
def update_distribution_users(pathname):
    if pathname == "/analyse-biais":
        ratings_per_user = ratings['user_id'].value_counts()
        fig = px.histogram(
            ratings_per_user, nbins=10, title="Distribution des évaluations par utilisateur",
            labels={'value': "Nombre d'évaluations par utilisateur", 'count': "Nombre d'utilisateurs"},
            log_y=False,
            text_auto=True
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(xaxis_title="Nombre d'évaluations", yaxis_title="Nombre d'utilisateurs")
        return fig
    return {}

@app.callback(
    Output("genre-distribution", "figure"),
    [Input("url", "pathname")]
)
def update_genre_distribution(pathname):
    if pathname == "/analyse-biais":
        genre_columns = [
            col for col in movies.columns 
            if col not in ["release_year", "movie_id"] and movies[col].dtype in ['int64', 'float64']]
        genre_distribution = movies[genre_columns].sum().sort_values(ascending=False)
        fig = px.bar(
            genre_distribution, title="Répartition des films par genre",
            labels={"index": "Genres", "value": "Nombre de films"},
            text_auto=True
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_layout(xaxis_title="Genres", yaxis_title="Nombre de films", xaxis_tickangle=45)
        return fig
    return {}

@app.callback(
    Output("gender-rating-mean", "figure"),
    [Input("url", "pathname")]
)
def update_gender_rating_mean(pathname):
    if pathname == "/analyse-biais":
        merged_data = ratings.merge(users, on='user_id')
        gender_rating = merged_data.groupby('gender')['rating'].mean()
        fig = px.bar(
            gender_rating, title="Note moyenne par genre",
            labels={"index": "Genre", "value": "Note moyenne"},
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black", marker_color='skyblue')
        fig.update_layout(xaxis_title="Genre", yaxis_title="Note moyenne", xaxis_tickangle=0)
        return fig
    return {}

@app.callback(
    Output("gender-rating-distribution", "figure"),
    [Input("url", "pathname")]
)
def update_gender_rating_distribution(pathname):
    if pathname == "/analyse-biais":
        merged_data = ratings.merge(users, on='user_id')
        fig = px.histogram(
            merged_data, x='rating', color='gender', nbins=5, title='Distribution des notes par genre',
            labels={'rating': 'Notes', 'count': 'Fréquence'}, opacity=0.7, barmode='overlay'
        )
        fig.update_layout(xaxis_title="Notes", yaxis_title="Fréquence")
        return fig
    return {}

@app.callback(
    Output("age-group-rating-mean", "figure"),
    [Input("url", "pathname")]
)
def update_age_group_rating_mean(pathname):
    if pathname == "/analyse-biais":
        merged_data = ratings.merge(users, on='user_id')
        age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
        age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        merged_data['age_group'] = pd.cut(merged_data['age'], bins=age_bins, labels=age_labels)
        age_rating = merged_data.groupby('age_group')['rating'].mean()
        fig = px.bar(
            age_rating, title="Note moyenne par groupe d'âge",
            labels={"index": "Groupe d'âge", "value": "Note moyenne"},
        )
        fig.update_traces(marker_line_width=1.5, marker_line_color="black", marker_color='orange')
        fig.update_layout(xaxis_title="Groupe d'âge", yaxis_title="Note moyenne", xaxis_tickangle=45)
        return fig
    return {}

@app.callback(
    Output("age-group-rating-distribution", "figure"),
    [Input("url", "pathname")]
)
def update_age_group_rating_distribution(pathname):
    if pathname == "/analyse-biais":
        merged_data = ratings.merge(users, on='user_id')
        age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
        age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        merged_data['age_group'] = pd.cut(merged_data['age'], bins=age_bins, labels=age_labels)
        fig = px.histogram(
            merged_data, x='rating', color='age_group', nbins=5, title="Distribution des notes par groupe d'âge",
            labels={'rating': 'Notes', 'count': 'Fréquence'}, opacity=0.7, barmode='overlay',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(xaxis_title="Notes", yaxis_title="Fréquence")
        return fig
    return {}

@app.callback(
    [Output("chi2-results", "children"),
     Output("heatmap-gender", "figure"),
     Output("heatmap-age", "figure")],
    [Input("url", "pathname")]
)
def update_chi2_results(pathname):
    if pathname == "/analyse-biais":
        merged_data = ratings.merge(users, on='user_id')

        # Test de Chi² pour les biais liés au genre
        gender_contingency = pd.crosstab(merged_data['gender'], merged_data['rating'])
        chi2_gender, p_gender, _, _ = chi2_contingency(gender_contingency)

        # Test de Chi² pour les biais liés aux groupes d'âge
        age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
        age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        merged_data['age_group'] = pd.cut(merged_data['age'], bins=age_bins, labels=age_labels)
        age_contingency = pd.crosstab(merged_data['age_group'], merged_data['rating'])
        chi2_age, p_age, _, _ = chi2_contingency(age_contingency)

        # Résultats
        chi2_results = [
            html.P(f"Test de Chi² pour les biais liés au genre : Chi² = {chi2_gender:.2f}, p-value = {p_gender:.4f}"),
            html.P(f"Test de Chi² pour les biais liés aux groupes d'âge : Chi² = {chi2_age:.2f}, p-value = {p_age:.4f}"),
        ]

        # Heatmap pour le genre
        fig_gender = px.imshow(
            gender_contingency.values,
            labels={"x": "Évaluations", "y": "Genre", "color": "Fréquence"},
            x=gender_contingency.columns,
            y=gender_contingency.index,
            color_continuous_scale="viridis",
            text_auto=True,
            title="Tableau de contingence : Genre vs. Évaluations"
        )

        # Heatmap pour l'âge
        fig_age = px.imshow(
            age_contingency.values,
            labels={"x": "Évaluations", "y": "Groupe d'âge", "color": "Fréquence"},
            x=age_contingency.columns,
            y=age_contingency.index,
            color_continuous_scale="viridis",
            text_auto=True,
            title="Tableau de contingence : Âge vs. Évaluations"
        )

        return chi2_results, fig_gender, fig_age

    return [], {}, {}

# Lancer l'application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render définit la variable d'environnement PORT
    app.run_server(host="0.0.0.0", port=port, debug=True)
