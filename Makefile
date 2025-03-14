# -----------------------------------------------------------------------------
# Set up
# -----------------------------------------------------------------------------

PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# commands
CMD := echo
CMD := sbatch submit1.sh
CMD := python

SIDS := 798
CONFIGS := configs/bridge-wm-ltm-config.py #configs/bridge-static-ridgecv-config.yml #configs/bridge-static-config.yml configs/bridge-groupcv-wmltm-config.yml configs/bridge-wm-config.yml configs/bridge-ltm-cl0-rr-config.yml configs/bridge-ltm-cl1-config.yml configs/bridge-ltm-cl1-rr-config.yml configs/bridge-wmltm-config.yml configs/bridge-wmltm-nofs-config.yml
run-encoding:
	mkdir -p logs
	for sid in $(SIDS); do \
		for config in $(CONFIGS); do \
			$(CMD) scripts/tfsenc_main.py \
				--config-file configs/config-tfs.yml configs/$$sid-config.yml configs/llama3-8b-retrieval-raw-fix-space-r.yml configs/lags2k-config-r.yml configs/subset-ref-recap-config.yml $$config; \
		done; \
	done;


run-encoding-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfsenc_main.py \
			--config-file configs/config.yml configs/$$sid-config.yml configs/glove-config-r.yml; \
	done;


run-erp:
	mkdir -p logs
	$(CMD) scripts/tfserp_main.py \
		--config-file configs/erp-config.yml configs/7170-config.yml configs/subset-ref-recap-config.yml;


run-erp-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfserp_main.py \
			--config-file configs/erp-config.yml configs/$$sid-config.yml; \
	done;
