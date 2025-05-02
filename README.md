# Projet GPU - Batch Merge Small (ENSAE 2025)

## 🔍 Objectif du projet

L'objectif de ce projet est de développer plusieurs versions d'un algorithme de fusion de tableaux triés (à la manière du Merge Path), en CUDA, pour de petits tableaux de taille maximale 1024.
Chaque question correspond à un niveau d'optimisation progressif de l'algorithme initial.

---

## ✅ Question 1 – `mergeSmall_k`

**Fusion séquentielle parallélisée sur un seul bloc**

* Fichier test : `src/test_mergeSmall.cu`
* Kernel : `mergeSmall_k`
* Fonctionne pour une unique paire `(A, B)` avec `|A| + |B| ≤ 1024`

**Commande :**

```bash
make run_small
```

**Exemple de sortie :**

```
Résultat de la fusion A + B :
1 2 3 4 5 6 7 8 9 10
```

---

## ✅ Question 2 – `mergeSmallBatch_k`

**Fusion par lot (batch)** de `N` couples de petits tableaux `(Ai, Bi)`

* Fichier test : `src/test_batchMerge.cu`
* Kernel : `mergeSmallBatch_k`
* Paramétrable par `d` (taille de chaque fusion)

**Commande :**

```bash
make run_batch
```

**Benchmark de base (10000 fusions) :**

```
d =  128 | Time ~ 0.34 ms
d =  256 | Time ~ 0.35 ms
d =  512 | Time ~ 0.73 ms
d = 1024 | Time ~ 1.45 ms
```

---

## ✅ Question 3 – Analyse & Optimisations

### 📊 Version 1 : `mergeSmallBatchShared_k`

* Optimisation par **mémoire partagée**
* Fichier : `src/benchmark_batchMerge_shared.cu`
* Kernel : `mergeSmallBatchShared_k`
* Rapide pour `d ≤ 256`, mais limite mémoire partagée

**Commande :**

```bash
make run_bench_shared
```

**Exemple :**

```
d =  128 | 0.37 ms
d =  256 | 0.26 ms
d =  512 | ❌ (pas assez de mémoire partagée)
d = 1024 | ❌
```

---

### 🌎 Version 2 : `mergeSmallOnePerBlock_k`

* Une fusion par bloc (robuste mais moins parallèle)
* Fichier : `src/benchmark_onePerBlock.cu`

**Commande :**

```bash
make run_bench_one
```

**Exemple :**

```
d = 128 | 37.9 ms
d = 256 | 0.31 ms
d = 512 | 0.68 ms
d = 1024 | 1.93 ms
```

---

### 🥇 Version 3 : `mergeSmallCoopWarp_k`

* Fusion coopérative : **1 warp (32 threads) par fusion**
* Excellente occupation GPU et performances stables
* Fichier : `src/benchmark_coopWarp.cu`

**Commande :**

```bash
make run_bench_coop
```

**Exemple :**

```
d = 128 | 43.2 ms (mauvaise occupation)
d = 256 | 0.05 ms
d = 512 | 0.05 ms
d = 1024 | 0.04 ms
```

---

## 🔄 Conclusion

Chaque version répond à des contraintes différentes :

* `mergeSmall_k` : base simple pour une seule fusion
* `mergeSmallBatch_k` : parallélisme massif pour petits `d`
* `mergeSmallBatchShared_k` : très rapide avec mémoire partagée (dès `d ≤ 256`)
* `mergeSmallOnePerBlock_k` : stable mais sous-optimale pour petits `d`
* `mergeSmallCoopWarp_k` : meilleure performance globale pour grands `d`

---

## 📁 Organisation des fichiers

```
include/
  merge_kernels.cuh          # Déclarations de tous les kernels
kernels/
  merge_kernels.cu           # Implémentations CUDA des kernels
src/
  test_mergeSmall.cu         # Test Q1
  test_batchMerge.cu         # Test Q2
  benchmark_batchMerge_shared.cu   # Q3 - mémoire partagée
  benchmark_onePerBlock.cu          # Q3 - 1 fusion par bloc
  benchmark_coopWarp.cu             # Q3 - warp coop
Makefile                     # Compilation des cibles + run
```

---

## ✉️ Lancement rapide

```bash
make run_small         # Q1
make run_batch         # Q2
make run_bench_shared  # Q3.1
make run_bench_one     # Q3.2
make run_bench_coop    # Q3.3
```

---

## 📄 Auteur

Projet réalisé dans le cadre du cours "Programmation en GPU" - ENSAE 2025
# Projet-GPU---Batch-Merge-Small