DATA_TRAIN_DIR=$1 # wic_train-en-en
targ_emb=$2 # dist_l1 or dist_l1ndotn
hs=$3 # 0
batch_norm=$4 # 1
bn1_track_stats=$5 # 1 if True, 0 if False
siamese=$6 #false
addings_params=${@:7}

eval_per_epoch=1
train_loss=crossentropy_loss
DATA_FT_DIR=rusemshift-data
ft_loss=mse_loss
pool=mean
train_ckpt=accuracy.nen-nen.score+accuracy.en-en.score

model_name=base
grad_acc_steps=8
train_epochs=30
ft_epochs=50
ft_save_by_score=spearman.dev.scd_2.score #spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score


OUTPUT_DIR=xlmr-${model_name}..data_train-${DATA_TRAIN_DIR}..siam-${siamese}..targ_emb-${targ_emb}..hs-${hs}..bn-${batch_norm}..bn1_track-${bn1_track_stats}..train_loss-${train_loss}..pool-${pool}..ckpt-${train_ckpt}

DATA_TRAIN_DIR=data_dumped_full/${DATA_TRAIN_DIR}
DATA_FT_DIR=data_dumped_full/${DATA_FT_DIR}

linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
train_scd=$([ "$ft_loss" == crossentropy_loss ] && echo "" || echo "--train_scd")
symmetric=$([ "$siamese" == "true" ] && echo "false" || echo "true")

echo OUTPUT_DIR = $OUTPUT_DIR

python run_model.py --do_train --do_validation --data_dir $DATA_TRAIN_DIR --output_dir $OUTPUT_DIR/train/ --gradient_accumulation_steps $grad_acc_steps \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $train_loss --linear_head $linhead --head_hidden_size $hs \
	--num_train_epochs $train_epochs --siamese $siamese --symmetric $symmetric --save_by_score $train_ckpt \
	--model_name xlm-roberta-$model_name --eval_per_epoch $eval_per_epoch --bn1_track_stats $bn1_track_stats --log_test_metrics $addings_params

#if DATA_FT_DIR != :
#	python run_model.py $train_scd --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir $OUTPUT_DIR/finetune/ --gradient_accumulation_steps $grad_acc_steps \
#		--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head $linhead --head_hidden_size $hs \
#		--num_train_epochs $ft_epochs --ckpt_path $OUTPUT_DIR/train/$train_ckpt/ --save_by_score $ft_save_by_score --siamese $siamese --symmetric $symmetric
