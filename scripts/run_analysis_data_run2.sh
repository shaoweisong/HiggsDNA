outdir="/eos/home-j/jiehan/parquet/nanov9/data"

python scripts/run_analysis.py --config "metadata/zgamma_data_run2.json" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/ --unretire_jobs --batch_system "local" --short
