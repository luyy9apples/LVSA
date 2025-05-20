DATADIR=../KG_data
EMB_PATH=../kge/FB15k_emb.pt

SAVE_PATH=../LVSA_model/model1-fb15k/

CUDA_VISIBLE_DEVICES=1 python main.py --do_test --data_path ${DATADIR}/FB15k-betae --checkpoint_path ${SAVE_PATH} -n 1 -b 100 -d 2000 --rank 2000 --cpu_num 8 --geo logicvsa --print_on_screen --test_batch_size 1 --cuda --emb_path ${EMB_PATH} --dataname FB15k --save_path ${SAVE_PATH} --tasks 2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up --valid_steps 1000 --log_steps 100
