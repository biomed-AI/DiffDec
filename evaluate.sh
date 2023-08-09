python reformat.py --samples_path /home/xiejunjie/project/DiffDec/sample_mols/crossdock_test_full/pockets_difflinker_full_crossdock_xiejunjie_pockets_difflinker_full_crossdock_fa_bs4_date03-02_time18-25-26.995870_epoch=199 \
                    --formatted_path ./formatted_single \
                    --true_smiles_path ./data/single/crossdock_test_table.csv

python -W ignore compute_metrics.py \
    /home/xiejunjie/project/DiffDec/formatted_single/crossdock_test_metric.smi

python -W ignore vina_preprocess.py \
    /home/xiejunjie/project/DiffDec/formatted_single/crossdock_test_metric.smi \
    /home/xiejunjie/project/DiffDec/formatted_single/crossdock_test_vina.smi 

python vina_docking.py --test_csv_path /home/xiejunjie/project/DiffDec/formatted_single/crossdock_test_vina.csv \
                    --results_pred_path formatted_single/result.pt \
                    --results_test_path formatted_single/result_testset.pt