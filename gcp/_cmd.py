from os import system


system("""gcloud ml-engine jobs submit training nothing \
--job-dir=gs://tsaplay-bucket/nothing \
--module-name=tsaplay.task \
--staging-bucket=gs://tsaplay-bucket/ \
--packages=/Users/seanbugeja/Code/Msc/dist/tsaplay-0.1.dev0.tar.gz \
--config=/Users/seanbugeja/Code/Msc/gcp/_config.json""")
