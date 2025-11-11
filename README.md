# Vigilance-Vague-Submersion
Pour la présentation du projet, merci de lire le fichier "Présentation du projet.pdf"

## Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Viictordvd/Vigilance-Vagues-Submersion.git
   cd Vigilance-Vagues-Submersion

2. **Installer les librairies**
   On recommande de créer un environnement virtuel puis d'installer les dépendances :
   ```bash
   pip install -r requirements.txt

## Explications
1. **Organisation du répertoire**
   - Dans le dossier "Références" se trouvent l'ensemble de la littérature scientifique que nous avons utilisé.
   - Dans le dossier "Anglais" se trouvent pour l'instant une banque de mots de vocabulaire ainsi qu'une rapide présentation du projet
   - A la source du répertoire se trouvent des notebooks jupyter et des fichiers python.

2. **Les notebooks**
   - Le notebook *Campbell_scalar_input* est le premier que nous avons écrit. Il a pour but de reproduire les résultats de la thèse de Elodie Perrin avec l'exemple jouet de la fonction Campbell2D. Cette fonction "simulateur" prend en entrée des données scalaire pour et recrache un scalaire.  
   Ce notebook fait appel à différentes fonctions que l'on retrouve dans les fichiers python *methods_for_scalar_input* et *Design of experiment*.

   - Le notebook *Campbell_functionnal_input* est le second que nous avons écrit. Il a pour objectif de reprendre le premier, mais cette fois avec des données d'entrée fonctionnelles. Notre simulateur prend maintenant en entrée des fonctions qui dépendent du temps et recrache toujours un scalaire.
   Ce notebook fait appel à différentes fonctions que l'on retrouve dans les fichiers python *ACPF*, *Gaussian_processes* et *validation_gp*.

   - Les autres notebooks ont été écrit pour nous aider à comprendre les outils que nous manipulons. Il ne serve à rien, à part aider à la compréhension. (ils méritent d'être mis à jour et commenté, on fera si on a le temps)


Auteurs : Paul Slisse et Victor Davodeau  
Sous la direction de : Pascale Noble, Olivier Roustant et Remy Baraille