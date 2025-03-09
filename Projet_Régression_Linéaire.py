# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1kYs6dYmUFCAwYeuC3XhPS-C7Q3tf4ZvM

L'objectif de ce projet est de construire un modèle de machine learning capable de prédire le prix de vente d'un bien immobilier en fonction de ses caractéristiques. Le dataset utilisé contient plusieurs variables liées aux propriétés des maisons, comme la superficie du terrain, la qualité globale de la construction, l'année de construction, la superficie du sous-sol fini, ainsi que des variables catégorielles liées au type de vente et aux conditions de vente.

### Importations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""### Données"""

# LECTURE CSV à éxécuter

# lecture à partir de Github pour pouvoir éxécuter le notebook sans se soucier d'importer le fichier csv
url = 'https://raw.githubusercontent.com/moncoachdata/MasterClass_DS_ML/main/AMES_Final_DF.csv'
df = pd.read_csv(url)

# Sinon lecture classique en important sur Google Colab ou depuis le chemin exact
# df = pd.read_csv("AMES_Final_DF.csv")

df.head()

df.info()

"""### Création de X et y
**Tâche : Le label ou variable cible que nous essayons de prédire est la colonne SalePrice. Séparez les données en Features X et Label y**.
"""

X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

"""### Fractionnement Entraînement Test
**Tâche : Utilisons scikit-learn pour séparer X et y en un ensemble d'entraînement et un ensemble de test. Comme nous utiliserons plus tard une stratégie de recherche par grille (grid search), fixez votre proportion de test à 10 %. Pour obtenir la même répartition des données que dans le notebook solutions, vous pouvez spécifier random_state = 101**.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

"""### Mise à l'échelle des données
**Tâche : Les features de l'ensemble de données ont une variété d'échelles et d'unités. Pour une performance de régression optimale, mettez à l'échelle les Features X. Prenons note de ce que vous utilisez pour .fit() et de ce que vous utilisez pour .transform()**.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

"""### Modèle
**Tâche : Nous allons utiliser un modèle Elastic Net. Créez une instance du modèle ElasticNet par défaut avec scikit-learn**.
"""

from sklearn.linear_model import ElasticNet

base_elastic_model = ElasticNet(max_iter=1000000)

"""**Tâche : Le modèle Elastic Net a deux paramètres principaux, alpha et le ratio L1 (l1_ratio). Créez un dictionnaire de grille de paramètres des valeurs pour l'Elastic Net. N'hésitons pas à jouer avec ces valeurs, gardez à l'esprit qu'elles peuvent ne pas correspondre exactement aux choix de la solution**."""

param_grid = {'alpha':[0.1,1,5,10,50,100],
              'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}

"""**Tâche : À l'aide de scikit-learn, créeons un objet GridSearchCV et exécutez un Grid Search pour les meilleurs paramètres de votre modèle en fonction de vos données d'entraînement mises à l'échelle. **"""

from sklearn.model_selection import GridSearchCV

# verbose (int) : Contrôle la verbosité, plus elle est élevée, plus il y a de messages
grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1)

grid_model.fit(scaled_X_train,y_train)

"""**Tâche : Afficher la meilleure combinaison de paramètres pour votre modèle**."""

grid_model.best_params_

"""### Évaluation du modèle
**Tâche : Évaluons les performances de votre modèle sur les 10% de données non vues auparavant (ensemble de test mis à l'échelle). Dans le notebook des solutions, nous avons obtenu une MAE de 14195 $\$$ et une RMSE de 20558 $\$$**.
"""

y_pred = grid_model.predict(scaled_X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_test,y_pred)

np.sqrt(mean_squared_error(y_test,y_pred))

np.mean(df['SalePrice'])