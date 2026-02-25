# Train RDE_DDG
#python run.py train --model_config ./config/models/train/RDE_DDG.yaml --data_config ./config/datasets/finetune/Trastuzumab.yaml --run_config ./config/runs/finetune_basic.yaml
# Train Unibind
#python run.py train --model_config ./config/models/train/UniBind.yaml --data_config ./config/datasets/train/Trastuzumab.yaml --run_config ./config/runs/train_basic.yaml
# UniAIR
python run.py train --model_config ./config/models/train/Prompt_DDG.yaml --data_config ./config/datasets/train/SKEMPIv2_promptDDG.yaml --run_config ./config/runs/train_basic.yaml

#python run.py dms --data_config ./config/datasets/downstream/dms.yaml --run_config ./config/runs/dms_basic.yaml --input_dir ./datasets/Zika/PDBs/5h37.pdb --chain A --valid_chains A,a,D,E --partners Aa_DE --period "10,20;40,45"


#python precompute_alignments.py ~/UniPPI/datasets/SKEMPIv2/fasta ./alignments --uniref90_database_path data/uniref90/uniref90.fasta \
#      --mgnify_database_path /dev/shm/data/mgnify/mgy_clusters_2022_05.fa \
#      --pdb_seqres_database_path /dev/shm/data/pdb_seqres/pdb_seqres.txt \
#      --uniref30_database_path /dev/shm/data/uniref30/UniRef30_2021_03 \
#      --uniprot_database_path /dev/shm/data/uniprot/uniprot.fasta \
#      --uniref90_database_path /dev/shm/data/uniref90/uniref90.fasta \
#      --bfd_database_path /dev/shm/data/small_bfd/bfd-first_non_consensus_sequences.fasta \
#      --cpus_per_task 8 \
#      --jackhmmer_binary_path ~/mambaforge/envs/openfold/bin/jackhmmer \
#      --hhblits_binary_path ~/mambaforge/envs/openfold/bin/hhblits \
#      --hmmsearch_binary_path ~/mambaforge/envs/openfold/bin/hmmsearch \
#      --hmmbuild_binary_path ~/mambaforge/envs/openfold/bin/hmmbuild \
#      --kalign_binary_path ~/mambaforge/envs/openfold/bin/kalign \
#      --use_small_bfd
