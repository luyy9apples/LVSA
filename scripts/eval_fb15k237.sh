DATADIR=../KG_data
EMB_PATH=../kge/FB15k-237_emb.pt

SAVE_PATH=../LVSA_model/model1-fb15k237/

CUDA_VISIBLE_DEVICES=3 python main.py --do_test --checkpoint_path ${SAVE_PATH} --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 100 -d 2000 --rank 2000 --cpu_num 8 --geo logicvsa --print_on_screen --test_batch_size 1 --cuda --emb_path ${EMB_PATH} --dataname FB15k-237 --save_path ${SAVE_PATH} --tasks 2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up --valid_steps 1000 --log_steps 100
