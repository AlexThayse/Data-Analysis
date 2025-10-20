import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#%% ACP

#Exploration des données
file_path = "leaf.csv"
leaf = pd.read_csv(file_path)

leaf.columns = [
    "Class", "Specimen Number", "Eccentricity", "Aspect Ratio",
    "Elongation", "Solidity", "Stochastic Convexity", "Isoperimetric Factor",
    "Maximal Indentation Depth", "Lobedness", "Average Intensity",
    "Average Contrast", "Smoothness", "Third moment", "Uniformity", "Entropy"
]

features = leaf.columns[2:]
X = leaf[features]
print("Taille de la matrice analysée:", leaf.shape)
print(leaf.head())



#Prétraitement des données
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

mean = np.mean(X_normalized) #avec le axis=0 si je veux la moyenne de chacune des variables
std_dev = np.std(X_normalized)
print(f"Moyenne après standardisation : {mean}")
print(f"Écart-type après standardisation : {std_dev}")

#Afficher la matrice de corrélation
print(X.corr(numeric_only=True))

#Afficher la heatmap de corrélation
plt.figure(figsize=(12, 10))
dataplot = sb.heatmap(X.corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.show()



#Appliquer l'ACP
pca = PCA(svd_solver='full')
pca_components = pca.fit_transform(X_normalized)
pca_df = pd.DataFrame(pca_components, columns=[f'Composante {i+1}' for i in range(pca_components.shape[1])])
print(pca_df)

eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(len(pca.explained_variance_))],
        "Valeur propre" : pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    },
    columns = ["Dimension", "Valeur propre", "% variance expliquée", "% cum. var. expliquée"]
)
print(eig)



#Visualisation pour le choix du nombre de composantes principales
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = np.cumsum(explained_variance)

data = {
    'Composante': [f'Composante {i+1}' for i in range(len(explained_variance))],
    'Variance expliquée': explained_variance,
    'Variance cumulée': explained_variance_cumulative
}
df = pd.DataFrame(data)

plt.figure(figsize=(12, 8))

# Barplot pour la variance expliquée
sb.barplot(x="Composante", y="Variance expliquée", data=df, color='lightseagreen')

# Ligne pour la variance cumulée
plt.plot(df['Composante'], df['Variance cumulée'], marker='o', color='red', label="Variance cumulée", linestyle='-', linewidth=2)

# Ajouter un seuil de sélection
plt.axhline(y=0.85, color='dimgray', linestyle="--", label="Seuil de 85% de variance")

plt.legend()
plt.xlabel('Composantes principales')
plt.ylabel('Variance expliquée')
plt.title('Scree Plot avec la variance expliquée et variance cumulée')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#Sur base du scree plot, on garde 3 composantes principales
print("ACP avec 3 composantes principales")
pca_3 = PCA(n_components=3)
pca3_components = pca_3.fit_transform(X)




print("Calcul des coordonnées des individus dans les 3 premières composantes principales")
X_pca_3 = pca.fit_transform(X_normalized)[:, :3]

explained_variance_1 = pca.explained_variance_ratio_[0] * 100  
explained_variance_2 = pca.explained_variance_ratio_[1] * 100  
explained_variance_3 = pca.explained_variance_ratio_[2] * 100

# Initialiser la figure pour les trois graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plan 1-2
axes[0].set_xlim(-6, 6)
axes[0].set_ylim(-6, 6)
for i in range(X_pca_3.shape[0]):
    axes[0].annotate(str(leaf.index[i]), (X_pca_3[i, 0], X_pca_3[i, 1]), fontsize=8)
axes[0].plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[0].plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
axes[0].set_title(f'Projection sur le plan PC1-PC2\n')
axes[0].set_xlabel(f"PC1 ({explained_variance_1:.2f}%)")
axes[0].set_ylabel(f"PC2 ({explained_variance_2:.2f}%)")

# Plan 1-3
axes[1].set_xlim(-6, 6)
axes[1].set_ylim(-6, 6)
for i in range(X_pca_3.shape[0]):
    axes[1].annotate(str(leaf.index[i]), (X_pca_3[i, 0], X_pca_3[i, 2]), fontsize=8)
axes[1].plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[1].plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
axes[1].set_title(f'Projection sur le plan PC1-PC3\n')
axes[1].set_xlabel(f"PC1 ({explained_variance_1:.2f}%)")
axes[1].set_ylabel(f"PC3 ({explained_variance_3:.2f}%)")

# Plan 2-3
axes[2].set_xlim(-6, 6)
axes[2].set_ylim(-6, 6)
for i in range(X_pca_3.shape[0]):
    axes[2].annotate(str(leaf.index[i]), (X_pca_3[i, 1], X_pca_3[i, 2]), fontsize=8)
axes[2].plot([-6, 6], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[2].plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
axes[2].set_title(f'Projection sur le plan PC2-PC3\n')
axes[2].set_xlabel(f"PC2 ({explained_variance_2:.2f}%)")
axes[2].set_ylabel(f"PC3 ({explained_variance_3:.2f}%)")

# Ajuster l'espacement entre les graphiques et afficher
plt.tight_layout()
plt.show()




print("Calcul de la contribution des individus à l'inertie totale (distance au carré à l'origine)")
di = np.sum(pca_components**2, axis=1)
print(pd.DataFrame({'ID': leaf.index, 'd_i': di}))


#Calcul de la qualité de la représentation des individus (cos²)
cos2 = X_pca_3**2
for j in range(3):  
    cos2[:, j] = cos2[:, j] / di
print(pd.DataFrame({'ID': leaf.index, 'COS2_1': cos2[:, 0], 'COS2_2': cos2[:, 1], 'COS3_3':cos2[:,2]}))





print("Calcul de la contribution des individus aux axes (CTR)")
ctr = X_pca_3**2
eigvals = pca.explained_variance_
n = X_pca_3.shape[0]
for j in range(3):  
    ctr[:, j] = ctr[:, j] / (n * eigvals[j])
print(pd.DataFrame({'ID': leaf.index, 'CTR_1': ctr[:, 0], 'CTR_2': ctr[:, 1], 'CTR_3': ctr[:,2]}))


results_df = pd.DataFrame({
    'ID': leaf.index,
    'Coord_PC1': X_pca_3[:, 0],
    'Coord_PC2': X_pca_3[:, 1],
    'Coord_PC3': X_pca_3[:, 2],
    'CTR_PC1': ctr[:, 0],
    'CTR_PC2': ctr[:, 1],
    'CTR_PC3': ctr[:, 2],
    'COS2_PC1': cos2[:, 0],
    'COS2_PC2': cos2[:, 1],
    'COS2_PC3': cos2[:, 2],
    'Sum_COS2': np.sum(cos2, axis=1)
})

# Affichage des 15 premiers individus
print(results_df.head(300))






# Calcul de la matrice des corrélations avec les axes PCA
sqrt_eigval = np.sqrt(pca.explained_variance_)

corvar = np.zeros((X.shape[1], pca.components_.shape[0]))
for k in range(pca.components_.shape[0]):
    corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]

# Créer une figure avec 3 sous-graphes pour les plans factoriels 1-2, 1-3 et 2-3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plan factoriel 1-2
axes[0].set_xlim(-1, 1)
axes[0].set_ylim(-1, 1)
axes[0].set_title('Plan factoriel 1-2')
for j in range(X.shape[1]):
    axes[0].annotate(X.columns[j], (corvar[j, 0], corvar[j, 1]))
    axes[0].quiver(0, 0, corvar[j, 0], corvar[j, 1], angles='xy', scale_units='xy', scale=1, color='grey', width=0.002, headwidth=10, headlength=8)
axes[0].plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[0].plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
axes[0].add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
axes[0].set_aspect('equal')
axes[0].set_xlabel(f"PC1 ({explained_variance_1:.2f}%)")
axes[0].set_ylabel(f"PC2 ({explained_variance_2:.2f}%)")

# Plan factoriel 1-3
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(-1, 1)
axes[1].set_title('Plan factoriel 1-3')
for j in range(X.shape[1]):
    axes[1].annotate(X.columns[j], (corvar[j, 0], corvar[j, 2])) 
    axes[1].quiver(0, 0, corvar[j, 0], corvar[j, 2], angles='xy', scale_units='xy', scale=1, color='grey', width=0.002, headwidth=10, headlength=8)
axes[1].plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[1].plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
axes[1].add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
axes[1].set_aspect('equal')
axes[1].set_xlabel(f"PC1 ({explained_variance_1:.2f}%)")
axes[1].set_ylabel(f"PC3 ({explained_variance_3:.2f}%)")

# Plan factoriel 2-3
axes[2].set_xlim(-1, 1)
axes[2].set_ylim(-1, 1)
axes[2].set_title('Plan factoriel 2-3')
for j in range(X.shape[1]):
    axes[2].annotate(X.columns[j], (corvar[j, 1], corvar[j, 2]))  
    axes[2].quiver(0, 0, corvar[j, 1], corvar[j, 2], angles='xy', scale_units='xy', scale=1, color='grey', width=0.002, headwidth=10, headlength=8)
axes[2].plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
axes[2].plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
axes[2].add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
axes[2].set_aspect('equal')
axes[2].set_xlabel(f"PC2 ({explained_variance_2:.2f}%)")
axes[2].set_ylabel(f"PC3 ({explained_variance_3:.2f}%)")

# Affichage
plt.tight_layout()
plt.show()




#Visualisation en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer les points dans l'espace 3D en fonction des 3 premières composantes principales
ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c='b', marker='o', s=50)

# Ajouter les annotations pour chaque individu
for i in range(X_pca_3.shape[0]):
    ax.text(X_pca_3[i, 0], X_pca_3[i, 1], X_pca_3[i, 2], str(leaf.index[i]), color='red', fontsize=8)

# Ajouter des étiquettes pour chaque axe
ax.set_xlabel(f'PC1 ({explained_variance_1:.2f}%)')
ax.set_ylabel(f'PC2 ({explained_variance_2:.2f}%)')
ax.set_zlabel(f'PC3 ({explained_variance_3:.2f}%)')

# Ajouter un titre
ax.set_title('Projection des individus dans les 3 premières composantes principales')

# Afficher le graphique
plt.tight_layout()
plt.show()

#%% ANOVA

import scipy.stats as stats 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import seaborn as sns
#%% Chargement du fichier CSV dans un DataFrame

file_path = "diet.csv"
df = pd.read_csv(file_path)
#%% Calcul de la perte de poids (poids avant - poids après 6 semaines)

df['perte_poids'] = df['preweight'] - df['weight6weeks']
df['perte_poids'] = df['perte_poids'].round(1)

#%% Affichage du DataFrame

print("DONNÉES :\n")
print(df.head())
print(df.shape)
print("===============================================================")

#%% Création d'une colonne combinée pour le genre et le régime

df['group'] = df['diet'] + "_" + df['gender']

#%% Création des groupes de données

groupes = [df[df['group'] == groupe]['perte_poids'] for groupe in df['group'].unique()]

print("GROUPE SUIVANT LE TYPE DE RÉGIME ET LE GENRE :")
print(df['group'].unique())
print("===============================================================")

#%% Histogrammes pour chaque groupe

# Définir le nombre de sous-graphes (par exemple, 3 lignes et 2 colonnes)
n = len(df['group'].unique())  # Le nombre de groupes
cols = 2  # Nombre de colonnes
rows = (n + 1) // cols  # Nombre de lignes

# Tracer un histogramme par groupe
for i, group in enumerate(df['group'].unique()):
    plt.figure(figsize=(8, 6))
    plt.plot(rows, cols, i+1)  # Définir le sous-graphe
    group_data = df[df['group'] == group]['perte_poids']
   
    # Tracer l'histogramme
    plt.hist(group_data, bins=10, alpha=0.7, label=group, color='skyblue', edgecolor='black')
   
    # Ajouter un titre et des étiquettes
    plt.title(f"Distribution de la perte de poids pour le groupe {group}", fontsize=12)
    plt.xlabel("Perte de poids (kg)", fontsize=10)
    plt.ylabel("Fréquence", fontsize=10)
    plt.legend()

    # Ajuster l'espacement entre les sous-graphes
    plt.tight_layout()
    plt.show()

#%%  Graphiques en boîtes à moustaches pour l'analyse visuelle des données
plt.figure(figsize=(10, 6))
plt.boxplot(groupes, labels=df['group'].unique(), patch_artist=True)
plt.title("Perte de poids par type de régime et genre", fontsize=14)
plt.xlabel("Type de régime et genre", fontsize=12)
plt.ylabel("Perte de poids (kg)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("===============================================================")
#%% Test de normalité avec QQ plot et test de Shapiro-Wilk

for groupe in df['group'].unique():

    # Récupération des données de perte de poids pour chaque groupe
    data = df[df['group'] == groupe]['perte_poids']
#%%QQ plot

    plt.figure(figsize=(8, 6))

    stats.probplot(data, dist="norm", plot=plt)

    plt.title(f"QQ plot pour le groupe {groupe}")
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Quantiles observés")

    plt.show()
#%%Test de Shapiro-Wilk
    stat, p_value = stats.shapiro(data)

    print(f"\nTest de Shapiro-Wilk pour le groupe {groupe} : Stat={stat:.3f}, p-value={p_value:.3f}")
    if p_value < 0.05:
        print(f"  Les données pour le groupe {groupe} ne suivent pas une distribution normale.")
    else:
        print(f"  Les données pour le groupe {groupe} suivent une distribution normale.")

print("===============================================================")

#%% Graphique des résidus
model = smf.ols('perte_poids ~ diet + gender', data=df).fit()

# Récupérer les résidus et les valeurs prédites
residuals = model.resid
predicted = model.fittedvalues

# Créer un tableau avec les indices des groupes (1, 2, 3, ..., 6)
groupes_indices = df['group'].unique()  # Liste des groupes

# Tracer les résidus pour chaque groupe
plt.figure(figsize=(8, 6))
for group in groupes_indices:
    group_data = df[df['group'] == group]
    group_residuals = group_data['perte_poids'] - model.predict(group_data)
    plt.scatter([group] * len(group_data), group_residuals, alpha=0.7, label=f"Groupe {group}")

plt.axhline(0, color='red', linestyle='--')  # Ligne horizontale pour 0
plt.title("Graphique des résidus vs groupe")
plt.xlabel("Groupes")
plt.ylabel("Résidus")
plt.grid(True)
plt.show()

#%%Test de Levene
levene_stat, levene_p_value = stats.levene(*groupes)

# Afficher les résultats
print("\nStatistique du test de Levene:", levene_stat)
print("P-value:", levene_p_value)

# Interprétation du test
alpha = 0.05
if levene_p_value < alpha:
    print("\nLes variances des groupes sont significativement différentes (H0 rejetée).\n")
else:
    print("\nLes variances des groupes ne sont pas significativement différentes (H0 acceptée).\n")
#%% Test de Bartlett pour tous les groupes

bartlett_stat, bartlett_p_value = stats.bartlett(*groupes)

print(f"Test de Bartlett : Stat={bartlett_stat:.3f}, p-value={bartlett_p_value:.3f}")

 

if bartlett_p_value < 0.05:

    print("\nLes variances entre les groupes sont significativement différentes (H0 rejetée).\n")

else:

    print("\nLes variances entre les groupes sont homogènes (H0 pas rejetée).\n")

#%%ANOVA à un facteur

print("\nANOVA à un facteur - Test de l'égalité des moyennes entre les groupes \n")
# Test ANOVA

stat, p_value = stats.f_oneway(*groupes)
# Affichage des résultats

print(f"Statistique de l'ANOVA (F) : {stat:.4f}")

print(f"p-valeur : {p_value:.4f}\n")
#%%Interprétation des résultats

if p_value < 0.05:

    print("-> Il y a une différence significative (p <= 0.05) entre les groupes.")

else:

    print("-> Il n'y a pas de différence significative (p > 0.05).")

print("===============================================================")  

#%%Appliquer le test de Tukey

tukey = pairwise_tukeyhsd(df['perte_poids'], df['group'], alpha=0.05)

#Affichage des résultats
print(tukey.summary())
print("===============================================================")  

#%%Modèle de l'ANOVA à deux facteurs avec interaction
model = smf.ols('perte_poids ~ C(diet) * C(gender)', data=df).fit()

# Calcul de l'ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Affichage des résultats de l'ANOVA
print("### ANOVA à deux facteurs avec interaction ###\n")
print(anova_table)

# Conclusion des tests sur les facteurs et leur interaction
if anova_table['PR(>F)']['C(diet)'] < 0.05:
    print("-> Le facteur 'type de régime' a un effet significatif sur la perte de poids.")
else:
    print("-> Le facteur 'type de régime' n'a pas d'effet significatif.")

 
if anova_table['PR(>F)']['C(gender)'] < 0.05:
    print("-> Le facteur 'genre' a un effet significatif sur la perte de poids.")
else:
    print("-> Le facteur 'genre' n'a pas d'effet significatif.")

 
if anova_table['PR(>F)']['C(diet):C(gender)'] < 0.05:
    print("-> Il y a une interaction significative entre le type de régime et le genre.")
else:
    print("-> Il n'y a pas d'interaction significative entre le type de régime et le genre.")

#%%Création du graphique d'interaction

plt.figure(figsize=(10, 6))

sns.pointplot(data=df, x='diet', y='perte_poids', hue='gender',

              markers=["o", "s"], linestyles=["-", "--"], dodge=True,

              palette="Set2")

plt.title("Interaction entre le type de régime et le genre sur la perte de poids", fontsize=14)

plt.xlabel("Type de régime", fontsize=12)

plt.ylabel("Perte de poids (kg)", fontsize=12)

plt.legend(title='Genre', loc='upper left')

plt.grid(True)

plt.show()

#%%ANOVA à contrast
def contraste(c, df):
    # Crée les groupes uniques pour la variable `group` qui contient des valeurs comme 'A_F', 'B_M', etc.
    groupes_uniques = df['group'].unique()
    print("Groupes uniques :", groupes_uniques)
   
    # Créer des vecteurs pour chaque groupe
    groupes = {g: df[df['group'] == g]['perte_poids'].values for g in groupes_uniques}
   
    # Afficher les groupes pour vérifier
    for g, data in groupes.items():
        print(f"Groupe {g}: {data}")

    #Nombre de groupes et nombre total de données
    r = len(groupes_uniques)  # Nombre de groupes
    n = len(df)  # Nombre total de données
   
    # Moyennes de chaque groupe et tailles des groupes
    mu = np.array([np.mean(groupes[g]) for g in groupes_uniques])
    ni = np.array([len(groupes[g]) for g in groupes_uniques])
   
    # Moyenne globale
    mu_tot = np.mean(df['perte_poids'])
   
    # Calcul de SCres
    SCres = sum(np.sum((groupes[g] - mu[i]) ** 2) for i, g in enumerate(groupes_uniques))
    MCres = SCres / (n - r)  # Moyenne des carrés résiduels
   
    # Calcul du contraste
    W = np.dot(c, mu)
    print(f"W (produit scalaire des moyennes et des contrastes) = {W}")
   
    # Calcul de la somme des carrés pour le contraste (MCw)
    tmp1 = np.dot(c, mu) ** 2
    tmp2 = np.dot(c, c)/n
    MCw = tmp1 / tmp2
   
    # Statistique F pour le contraste
    F_contraste = MCw / MCres
    print(f"Statistique F pour le contraste = {F_contraste}")
   
    # Calcul de la valeur critique de F
    f_critique_contraste = stats.f.ppf(0.95, 1, n - r)
    print(f"Valeur critique de F pour le contraste : {f_critique_contraste:.4f}")
   
    # Vérification du contraste
    if F_contraste > f_critique_contraste:
        print(f"On rejette H0 pour ce contraste, il y a une différence significative.")
    else:
        print(f"On ne rejette pas H0 pour ce contraste, il n'y a pas de différence significative.")
   
    return F_contraste, f_critique_contraste


#Contrast : Interaction entre les genres pour les régimes (C vs A/B)
print("### Interaction entre les genres pour les régimes (C vs A/B)\n")
c = np.array([0.5,-0.5 ,-0.5 ,1 ,0.5 ,-1] )  
F_contraste, f_critique = contraste(c, df)
print("===============================================================")    


# Contraste 1 : A_F vs C_F
c_1 = [0, 1, 0, -1, 0, 0]  # Comparer A_F et C_F
print("### Comparaison A_F vs C_F")
F_contraste, f_critique = contraste(c_1, df)
print("===============================================================")  


# Contraste 2 : B_F vs C_F
c_2 = [0, 0, 1, -1, 0, 0]  # Comparer B_F et C_F
print("### Comparaison B_F vs C_F")
F_contraste, f_critique = contraste(c_2, df)
print("===============================================================")  


# Contraste 3 : A_F vs A_M
c_3 = [0, 1, 0, 0, -1, 0]  # Comparer A_F et A_M
print("### Comparaison A_F vs A_M")
F_contraste, f_critique = contraste(c_3, df)
print("===============================================================")  


# Contraste 4 : B_M vs B_F
c_4 = [-1, 0, 1, 0, 0, 0]  # Comparer B_M et B_F
print("### Comparaison B_M vs B_F")
F_contraste, f_critique = contraste(c_4, df)
print("===============================================================")  

# Contraste 5 : C_M vs C_F
c_5 = [0, 0, 0, 1, 0, -1]  # Comparer C_M et C_F
print("### Comparaison C_M vs C_F")
F_contraste, f_critique = contraste(c_5, df)
print("===============================================================")

# Contraste 6 : A_F vs B_F
c_6 = [0, 1, -1, 0, 0, 0]
print("### Comparaison A_F vs B_F")
F_contraste, f_critique = contraste(c_6, df)
print("===============================================================")

# Contraste 7 : A_M vs B_M
c_7 = [1, 0, 0, 0, -1, 0]
print("### Comparaison A_M vs B_M")
F_contraste, f_critique = contraste(c_7, df)
print("===============================================================")

# Contraste 8 : C_M vs B_M
c_8 = [1, 0, 0, 0, 0, -1]
print("### Comparaison C_M vs B_M")
F_contraste, f_critique = contraste(c_8, df)
print("===============================================================")
