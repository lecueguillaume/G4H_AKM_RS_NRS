# G4H_AKM_RS_NRS

This is the repository for the project on AKM, recommendation system and neural recommendation system.

## Introduction

Au milieu de l’année 2023, des chercheurs danois ont publié un article marquant présentant des résultats remarquables. En exploitant des données de santé couvrant l’ensemble de la population du Danemark sur plusieurs années, combinées aux avancées significatives du **Deep Learning**, notamment grâce à l’architecture **Transformer**, ils ont réussi à créer un **embedding** de la population danoise reflétant l’état de santé des individus. L’un des résultats les plus frappants de cette étude est une visualisation en deux dimensions de cet embedding, où les individus sont répartis selon leur état de santé : d’un côté, ceux très susceptibles de décéder, et de l’autre, ceux en excellente santé, avec une zone intermédiaire représentant des états de santé plus nuancés.

Mon stage s’inscrit dans le cadre du projet **Graph4Health** mené au **CREST**, qui vise à reproduire ce type de résultats à une échelle encore plus large, à partir de données françaises, tout en adoptant une approche à la fois économique et économétrique. Ce projet ambitionne d’explorer et de mieux comprendre les liens entre les données de santé et d’autres variables économiques. Une description détaillée du projet est accessible via ce lien : [https://faculty.crest.fr/pchone/wp-content/uploads/sites/11/Graph4Health_pub.pdf].

Le présent rapport s’articule autour de plusieurs étapes clés qui reflètent la progression de mon travail sur l’analyse des **embedding spaces** générés à partir de graphes bipartites, dans le but d’étudier leur robustesse, leur visualisation et leur interprétation. En raison de l’inaccessibilité des données de santé, liée à des restrictions strictes de confidentialité, mon travail s’est concentré sur des données simulées à l’aide du modèle **AKM (Abowd, Kramarz, and Margolis)**. L’estimation des embeddings a été réalisée à l’aide de modèles généralement employés dans les systèmes de recommandation, en particulier la **Matrix Factorization** et un **Neural Recommendation System (NRS)**.

## Terminologie

Certains termes seront utilisés de manière interchangeable :  
- Les notions d’**embedding** et d’**effet fixe** renverront à la même idée.  
- Le terme effet fixe sera généralement employé dans le cadre de la génération, tandis que le terme embedding sera préféré dans le cadre de l’estimation.

Par ailleurs, les notations `α` et `ψ` désigneront respectivement les effets fixes des patients et des docteurs. Lorsqu’un accent circonflexe (chapeau, ^) sera ajouté au-dessus de ces symboles, il fera référence à l’estimation des embeddings dans le modèle, ou à l’estimation des effets fixes.

## Informations pratiques

Mes Notebooks ont été écrit dans cet ordre: 1) Clustering_AKM_MF_1D.ipynb  2) Clustering_AKM_MF_multidim.ipynb  3) Clustering_NRS.ipynb  4) Heatmap_AMI.ipynb
Bien sûr, ils peuvent s'exécuter indépendament des autres.

N'hésitez pas à me contacter sur Linkedin ou par mail à vincent.gimenes@ensae.fr en cas de problème avec mon code. 
