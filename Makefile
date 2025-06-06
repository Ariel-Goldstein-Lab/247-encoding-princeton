# -----------------------------------------------------------------------------
# Set up
# -----------------------------------------------------------------------------

PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-embedding/results/* data/
	# ln -fs /scratch/gpfs/kw1166/247/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# commands
SID := 625
EMB := gpt2-xl#gemma-2-2b#gpt2-xl#glove-nm
REGULARIZATION := none
CMD := python#sbatch --job-name=enc_$(SID)-$(EMB)-r_$(REGULARIZATION) submit.sh
# {echo | python | sbatch --job-name=$(SID)-$(EMB)-r_$(REGULARIZATION) submit.sh}

# Define the config path based on RIDGE value
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


SIDS:= 625 676 7170 798
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
