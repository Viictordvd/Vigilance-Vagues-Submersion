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
   - Le dossier "**Références**" contient l'ensemble de la littérature scientifique que nous avons utilisée.
   - Le dossier "**Anglais**" contient pour l'instant une banque de mots de vocabulaire ainsi qu'une rapide présentation du projet.
   - A la racine du répertoire se trouvent des notebooks Jupyter et des fichiers Python.
   - Le dossier "**data**"" contient les plans d'expériences utilisés par nos notebook
   - Les dossiers "**data_malo_GP**" et "**data_malo_GP_temporal**" contiennent des simulations TOLOSA. (merci de lire data_malo_GP/Descriptif_simu_tolosa_processus_gaussiens.pdf pour plus d'infos) 
   - Le dossier "**output-GIF**" contient des cartes de submersions en fonction du temps.

2. **Les notebooks**
Les notebooks font appel à différentes fonctions que l'on retrouve dans les fichiers python *model_class*, *ACPF*, *Gaussian_processes* et *validation_gp*.

   - Le notebook *Campbell_scalar_input* est le premier que nous avons écrit. Il a pour but de reproduire les résultats de la thèse de Elodie Perrin avec l'exemple jouet de la fonction Campbell2D. Cette fonction "simulateur" prend en entrée des données scalaire pour et recrache un scalaire.  

   - Le notebook *Campbell_functionnal_input* est le second que nous avons écrit. Il a pour objectif de reprendre le premier, mais cette fois avec des données d'entrée fonctionnelles. Notre simulateur prend maintenant en entrée des fonctions qui dépendent du temps et recrache toujours un scalaire. 

   - Le notebook *Campbell_functionnal_output* a pour objectif de sortir des cartes de submersion qui dépendent du temps, tout en gardant des données d'entrée fonctionnelles.
   
   - Le notebook *DoE Functionnal* permet de générer des plans d'expériences pour les deux notebooks précédents.

   - Le notebook *Saint_Malo_scalar_input* est très similaire au premier, car il a pour objectif d'appliquer ce qu'on a fait au cas réel de la baie de Saint_Malo
   
   - Les autres notebooks ont été écrits pour nous aider à comprendre les outils que nous manipulons. Ils n’ont pas d’utilité directe, mais facilitent la compréhension (ils méritent d'être mis à jour et commenté, on fera si on a le temps)

Auteurs : Paul Slisse et Victor Davodeau  
Sous la direction de : Pascale Noble, Olivier Roustant et Remy Baraille
