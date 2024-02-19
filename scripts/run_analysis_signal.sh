outdir="/afs/cern.ch/work/z/zewang/private/HZGamma/signal2"

python scripts/run_analysis.py --sample_list "ggH_M125" --config "metadata/zgamma_signal.json" --log-level "DEBUG" --n_cores 5 --output_dir /eos/user/s/shsong/HiggsDNA/hzgtest --unretire_jobs --short --batch_system "local"
#python scripts/convert_parquet_to_root.py --source $outdir/ggH_M125_2017/merged_nominal.parquet --target $outdir/merged.root --type mc --log DEBUG --process ggh --notag
