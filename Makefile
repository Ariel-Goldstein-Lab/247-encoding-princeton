# -----------------------------------------------------------------------------
# Set up
# -----------------------------------------------------------------------------

PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-embedding/results/* data/
	# ln -fs /scratch/gpfs/kw1166/247/247-pickling/results/* data/
	# ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# commands
SID := 661
EMB := gemma-scope-2b-pt-res-canonical#gemma-2-2b#gpt2-xl#glove-nm
REGULARIZATION := lasso#ridge

TIME := 1:00:00
MEM := 80GB
GPUS := 1

CMD := sbatch --time=$(TIME) --mem=$(MEM) --gres=gpu:$(GPUS) --job-name=enc_$(SID)-$(EMB)-r_$(REGULARIZATION) submit.sh
# {echo | python | sbatch --time=$(TIME) --mem=$(MEM) --gres=gpu:$(GPUS) --job-name=$(SID)-$(EMB)-r_$(REGULARIZATION) submit.sh}
# --dependency=afterok:$JOB_ID

# Define the config path based on REGULARIZATION value
ifeq ($(REGULARIZATION),ridge)
    EMB_CONFIG_PATH := $(EMB)-config-r.yml
else ifeq ($(REGULARIZATION),lasso)
    EMB_CONFIG_PATH := $(EMB)-config-l.yml
else
    EMB_CONFIG_PATH := $(EMB)-config.yml
endif

run-encoding:
	mkdir -p logs
	$(CMD) scripts/tfsenc_main.py \
		--config-file config.yml $(SID)-config.yml $(EMB_CONFIG_PATH)


SIDS:= 661 662 717 723 741 742 763 798
#{625 676 7170 798 | 661 662 717 723 741 742 743 763 798 | 777}
run-encoding-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfsenc_main.py \
			--config-file config.yml $$sid-config.yml $(EMB_CONFIG_PATH); \
	done;


run-erp:
	mkdir -p logs
	$(CMD) scripts/tfserp_main.py \
		--config-file configs/erp-config.yml configs/625-config.yml


run-erp-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfserp_main.py \
			--config-file configs/erp-config.yml configs/$$sid-config.yml; \
	done;
