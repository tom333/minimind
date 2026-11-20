#!/usr/bin/env bash
# Profilage par composant, à lancer quand le GPU est libre : attend la fin du run en cours, abandonne à 6h15
# (veille du matin à 6h40). Résultat dans out/profil_composants.txt.
cd "$(dirname "$0")/../trainer"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2500 ]; do
  [ "$(date +%H%M)" -ge 0615 ] && [ "$(date +%H%M)" -lt 1200 ] && { echo "GPU occupé jusqu'à 6h15 : abandon" > ../out/profil_composants.txt; exit 0; }
  sleep 60
done
{ date; for v in 16000 6400; do ../.venv/bin/python profile_components.py --vocab_size $v 2>&1 | grep -v -i "warn\|W11"; echo; done; } > ../out/profil_composants.txt 2>&1
