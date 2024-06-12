python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "ggH_customised_test" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/Run2signalWPL --unretire_jobs --short --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "ggH_customised_test" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/Run2signalWP90 --unretire_jobs --short --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data.json" --sample_list "ggH_customised_test" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/Run2signalHZZ --unretire_jobs --short --batch_system "local"


python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --log-level "DEBUG" --n_cores 10 --output_dir "/eos/user/s/shsong/HiggsDNA/Run3_Data" --sample_list "Data_FG","Data_CD","Data_E" --unretire_jobs --merge_outputs --batch_system "condor"
python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --sample_list "gghzg_run3" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/GluGluHToZG_run3wp80 --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --sample_list "gghzg_run3v6" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/GluGluHToZG_run3v6 --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --sample_list "gghzg_run3" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/GluGluHToZG_run3 --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --sample_list "gghzg_run3" --log-level "DEBUG" --n_cores 10 --output_dir /eos/user/s/shsong/HiggsDNA/GluGluHToZG_run3_WP90 --unretire_jobs --batch_system "local"
python scripts/run_analysis.py --config "metadata/zgamma_data_run3HZZ.json" --log-level "DEBUG" --n_cores 10 --output_dir "/eos/user/s/shsong/HiggsDNA/Run3_DY" --unretire_jobs --merge_outputs --sample_list "DYJetsToLL_Run3"  --batch_system "local"
