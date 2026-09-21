# Assets du mannequin

Ces fichiers sont **dérivés** du maillage anatomique libre de MakeHuman :

- maillage de base et cibles de morphologie : *MakeHuman*, publiés en **CC0 1.0**
  par Data Collection AB (https://github.com/makehumancommunity/makehuman,
  fichiers `LICENSE.ASSETS.md` et `data/3dobjs/base.obj`) ;
- squelette et poids de peau : `data/rigs/default.mhskel` et `default_weights.mhw`,
  également **CC0** (`© 2021 Data Collection AB, Joel Palmius, Jonas Hauquier`).

Le CC0 ne demande aucune attribution : elle est ici par courtoisie, et pour que
l'origine des données reste claire. Régénérer ces fichiers :

    python tools/build_mannequin_assets.py /chemin/vers/makehuman/data
