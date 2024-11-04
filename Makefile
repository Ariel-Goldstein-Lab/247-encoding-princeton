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
CMD := python
CMD := sbatch submit1.sh

CLS := 256
SIDS := 798
run-encoding:
	mkdir -p logs
	for sid in $(SIDS); do \
		for cl in $(CLS); do \
			$(CMD) scripts/tfsenc_main.py \
				--config-file configs/config-tfs.yml configs/$$sid-config.yml configs/autocompress-manual-r.yml configs/layer16-config.yml configs/cl$$cl-config.yml configs/lags200k-config-r.yml configs/save-preds-config.yml configs/subset-ref-recap-config.yml; \
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
		--config-file configs/erp-config.yml configs/625-config.yml


run-erp-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfserp_main.py \
			--config-file configs/erp-config.yml configs/$$sid-config.yml; \
	done;
