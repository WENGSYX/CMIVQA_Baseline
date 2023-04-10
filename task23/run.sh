shape=base \
seed=42 \
epochs=32 \
max_seq_length=1300 \
lr=1e-5 \
num_workers=0 \
weight_decay=1e-6 \
device=0 \


python main.py --device $device \
	--seed $seed \
	--maxlen $max_seq_length \
	--epochs $epochs \
	--lr $lr \
	--num_workers $num_workers \
	--weight_decay $weight_decay \


