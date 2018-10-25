from os import system


system("""gcloud ml-engine jobs submit training sum_steps_100_save_chkpt_steps250_metadata_rep3 \
--job-dir=gs://tsaplay-bucket/sum_steps_100_save_chkpt_steps250_metadata_rep3 \
--module-name=tsaplay.task \
--staging-bucket=gs://tsaplay-bucket/ \
--packages=/Users/seanbugeja/Code/Msc/dist/tsaplay-0.1.dev0.tar.gz \
--config=/Users/seanbugeja/Code/Msc/gcp/_config.json """)
