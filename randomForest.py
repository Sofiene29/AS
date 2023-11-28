import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Charger les données
features_file = 'acsincome_ca_features.csv'
labels_file = 'acsincome_ca_labels.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

# Mélanger les données
X_all, y_all = shuffle(features, labels, random_state=1)

# Sélectionner une fraction du dataset (par exemple, 10%)
num_samples = int(len(X_all) * 0.1)
X, y = X_all[:num_samples], y_all[:num_samples]

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train set et test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialisation du modèle Random Forest
rf_model = RandomForestClassifier(random_state=1)

# Paramètres pour la grille de recherche
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Mise en place de GridSearchCV pour le modèle Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

# Recherche des meilleurs paramètres
grid_search_rf.fit(X_train, y_train.values.ravel())

# Affichage des meilleurs paramètres trouvés
print("Meilleurs paramètres trouvés pour Random Forest : ", grid_search_rf.best_params_)

# Utilisation du meilleur modèle issu de GridSearchCV
best_rf_model = grid_search_rf.best_estimator_

# Prédiction sur l'ensemble de test avec le meilleur modèle
y_pred_best_rf = best_rf_model.predict(X_test)

# Évaluation du meilleur modèle
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
report_best_rf = classification_report(y_test, y_pred_best_rf)
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_best_rf)

# Affichage des performances du meilleur modèle
print(f"Accuracy pour le meilleur modèle Random Forest: {accuracy_best_rf}")
print(f"Rapport de classification pour le meilleur modèle Random Forest:\n{report_best_rf}")
print(f"Matrice de confusion pour le meilleur modèle Random Forest:\n{conf_matrix_best_rf}")
