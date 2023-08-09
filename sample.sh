CUDA_VISIBLE_DEVICES=0 python -W ignore sample_single.py \
                 --checkpoint ckpt/diffdec_single.ckpt \
                 --samples sample_mols \
                 --data data/single \
                 --prefix crossdock_test_full \
                 --n_samples 100 \
                 --device cuda:0
