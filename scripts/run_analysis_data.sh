outdir="/afs/cern.ch/work/z/zewang/private/HZGamma/data_run2_2017_v1"

python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "DoubleEG_Run2017B","DoubleEG_Run2017C","DoubleEG_Run2017D","DoubleEG_Run2017E","DoubleEG_Run2017F" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsZGammaAna/DoubleEGData_2017_hzzID --unretire_jobs --batch_system "condor"
python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "DoubleEG_Run2017B","DoubleEG_Run2017C","DoubleEG_Run2017D","DoubleEG_Run2017E","DoubleEG_Run2017F" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsZGammaAna/DoubleEGData_2017_WPLID --unretire_jobs --batch_system "condor"

python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "ggH_customised_test" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsZGammaAna/signal --unretire_jobs --short --batch_system "local"
