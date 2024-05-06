outdir="/afs/cern.ch/work/z/zewang/private/HZGamma/signal2"

python scripts/run_analysis.py --sample_list "ggH_customised_test" --config "metadata/zgamma_data.json" --log-level "DEBUG" --n_cores 5 --output_dir /eos/user/s/shsong/HiggsDNA/hzgsigwpl --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --sample_list "ggH_customised_test" --config "metadata/zgamma_data.json" --log-level "DEBUG" --n_cores 5 --output_dir /eos/user/s/shsong/HiggsDNA/hzgsighzz --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --sample_list "ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8" --config "metadata/zgamma_data.json" --log-level "DEBUG" --n_cores 5 --output_dir /eos/user/s/shsong/HiggsDNA/smzgwpl --unretire_jobs --short --batch_system "local"
python scripts/run_analysis.py --sample_list "ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8" --config "metadata/zgamma_data.json" --log-level "DEBUG" --n_cores 5 --output_dir /eos/user/s/shsong/HiggsDNA/smzghzz --unretire_jobs --short --batch_system "local"
#python scripts/convert_parquet_to_root.py --source $outdir/ggH_M125_2017/merged_nominal.parquet --target $outdir/merged.root --type mc --log DEBUG --process ggh --notag
