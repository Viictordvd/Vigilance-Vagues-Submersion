Les notebooks font appel à différentes fonctions que l'on retrouve dans les fichiers python *model_class*, *ACPF*, *Gaussian_processes* et *validation_gp* qui sont dans le repertoire **Methodes**.
   - Le notebook *Campbell_scalar_input* est le premier que nous avons écrit. Il a pour but de reproduire les résultats de la thèse de Elodie Perrin avec l'exemple jouet de la fonction Campbell2D. Cette fonction "simulateur" prend en entrée des données scalaire pour et recrache un scalaire.  

   - Le notebook *Campbell_functionnal_input* est le second que nous avons écrit. Il a pour objectif de reprendre le premier, mais cette fois avec des données d'entrée fonctionnelles. Notre simulateur prend maintenant en entrée des fonctions qui dépendent du temps et recrache toujours un scalaire. 

   - Le notebook *Campbell_functionnal_output* a pour objectif de sortir des cartes de submersion qui dépendent du temps, tout en gardant des données d'entrée fonctionnelles.
   
   - Le notebook *DoE Functionnal* permet de générer des plans d'expériences pour les deux notebooks précédents.