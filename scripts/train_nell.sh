DATADIR=../KG_data
EMB_PATH=../kge/NELL_emb.pt

SAVE_PATH=../LVSA_model/model0-nell/

CUDA_VISIBLE_DEVICES=0 python -u main.py --do_train --do_valid --data_path ${DATADIR}/NELL-betae -n 1 -b 2000 -d 2000 --rank 2000 --cpu_num 8 --geo logicvsa --print_on_screen --test_batch_size 1 --cuda --emb_path ${EMB_PATH} --dataname NELL --save_path ${SAVE_PATH} --tasks 2p.3p --valid_steps 1000 --log_steps 100 -lr 0.0005 --save_checkpoint_steps 1000 --max_steps 10000 --warm_up_steps 5000

SAVE_PATH=../LVSA_model/model1-nell/ 
CHECK_PATH=../LVSA_model/model0-nell/ 

CUDA_VISIBLE_DEVICES=0 python -u main.py --do_train --do_valid --data_path ${DATADIR}/NELL-betae -n 1 -b 2000 -d 2000 --rank 2000 --cpu_num 0 --geo logicvsa --print_on_screen --test_batch_size 1 --cuda  --emb_path ${EMB_PATH} --dataname NELL --save_path ${SAVE_PATH} --checkpoint_path ${CHECK_PATH}  --tasks 2in --valid_steps 1000 --log_steps 100 -lr 0.0005 --save_checkpoint_steps 1000 --max_steps 10000 --warm_up_steps 5000