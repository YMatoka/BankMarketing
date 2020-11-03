# Importation des packages
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

# Fonction de nettoyage des données
bank = pd.read_csv(r"/home/yann/Documents/Python_Dev/Bank marketing project/data_folder/bank-full.csv", delimiter = ";")

# Remplacement des valeurs
bank = bank.replace(['single', 'married', 'divorced', 'unknown', 
                      'primary','secondary','tertiary', 'yes', 'no', -1, "success", "failure", "other"], 
                      [1,2,3,0,1,2,3,1,0,0,1,0,2])


# Replacement jobs
bank = bank.replace(["management","technician", "entrepreneur", "blue-collar", "unknown","retired", 
                     "admin.", "services" ,"self-employed", "unemployed","housemaid","student"],
                      [0,1,2,3,4,5,6,7,8,9,10,11])

# Drop des columns
bank = bank.drop(columns=["contact","day","month","default"])


# Transforme notre tableau de données
scaler = MinMaxScaler()
bank = scaler.fit_transform(bank)

# Transforme le numpy en dataframe
bank = pd.DataFrame(bank, columns=["age","job","marital","education",
                                   "balance","housing","loan","duration", 
                                   "campaign", "pdays", "previous","poutcome","y"])


# Shape montre le nombre de colonnes
#bank.shape()

# Lecture des données via pandas
#bank.describe()

# Axes et variables
x,y = bank, bank.y

# Nom des colonnes
names = bank.columns

# nuage de point et data vis
nuage_point = pd.plotting.scatter_matrix(bank, figsize =  (16, 16))


# Corrélation des données 
corr = bank.corr() 

# Plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#corr

# Split des données
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, shuffle = True, random_state = 42)


# Création du model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)


# Fitting des prédictions avec exercice et résultat
knn.fit(x_train, y_train)


# Récupération des exercices et application des prédictions
y_pred = knn.predict(x_test)


# Résultat des prédictions
#y_pred

#Précision des prédictions en pourcentage
accuracy_score(y_test, y_pred)

# Résultat
#y_test


# Arbre de classification
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

# Affichage de l'arbre 
#plt.close()
#plt.figure(figsize = (12, 12))
#plot_tree(tree)
#plt.show()



#route menant à la page d'acceuil
@app.route("/")
def home():
    return "The Bank Marketing datasets"

@app.route("/dfjson")
def dfjson():
    """
    return a json representation of the dataframe
    """
    #df_json = bank.to_json(orient="records")
    df_list = bank.values.tolist()
    JSONP_data = jsonify(df_list)
    #parsed = json.loads(df_json)

    return JSONP_data


if __name__ == "__main__":
    app.run(debug=True)

