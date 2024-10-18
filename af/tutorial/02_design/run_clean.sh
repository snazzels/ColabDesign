#! /bin/bash

./scripts_clean/iptm.py

cd ranked_top
mkdir top
mv *pdb top/

#./scripts_clean/01_split.sh

./../scripts_clean/02_add_ter.py top/*pdb

./../scripts_clean/03_chain_id.py top/*cleaned.pdb

cd cleaned_top
./../../scripts_clean/cis_or_trans.py
cp -r trans ../../best_top

cd ../../
mv ranked_top trash

cd best_top
./../scripts_clean/sort_cys.py

