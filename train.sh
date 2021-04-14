DATA_TRAIN_DIR=wic
train_loss=crossentropy_loss
DATA_FT_DIR=rusemshift-data
ft_loss=mse_loss
pool=mean
batch_norm=0
train_ckpt=accuracy.nen-nen.score

grad_acc_steps=64
train_epochs=33
ft_epochs=50
ft_save_by_score=spearman.dev.scd_2.score #spearman.dev.scd_1.score+spearman.dev.scd_2.score+spearman.dev.scd_1.wordwise.score+spearman.dev.scd_2.wordwise.score

siamese=true
targ_emb=concat # dist_l1 or dist_l1ndotn
hs=-1

OUTPUT_DIR=xlmrlarge-siamese-data_train-${DATA_TRAIN_DIR}-data_ft-${DATA_FT_DIR}-train_loss-${train_loss}-ft_loss-${ft_loss}-pool-${pool}-targ_emb-${targ_emb}-hs-${hs}-bn-${batch_norm}-ckpt-${train_ckpt}

DATA_TRAIN_DIR=data_dumped_full/${DATA_TRAIN_DIR}
DATA_FT_DIR=data_dumped_full/${DATA_FT_DIR}

linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")
train_scd=$([ "$ft_loss" == crossentropy_loss ] && echo "" || echo "--train_scd")
symmetric=$([ "$siamese" == "true" ] && echo "false" || echo "true")

echo OUTPUT_DIR = $OUTPUT_DIR

python run_model.py --do_train --do_validation --data_dir $DATA_TRAIN_DIR --output_dir $OUTPUT_DIR/train/ --gradient_accumulation_steps $grad_acc_steps \
	--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $train_loss --linear_head $linhead --head_hidden_size $hs \
	--num_train_epochs $train_epochs --siamese $siamese --symmetric $symmetric --save_by_score $train_ckpt --eval_per_epoch 2 \
	--model_name "xlm-roberta-base"

#if DATA_FT_DIR != :
#	python run_model.py $train_scd --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir $OUTPUT_DIR/finetune/ --gradient_accumulation_steps $grad_acc_steps \
#		--pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head $linhead --head_hidden_size $hs \
#		--num_train_epochs $ft_epochs --ckpt_path $OUTPUT_DIR/train/$train_ckpt/ --save_by_score $ft_save_by_score --siamese $siamese --symmetric $symmetric
