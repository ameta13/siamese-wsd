data_train=wic_train-en-en
hs=-1

for targ_emb in comb_dmn comb_admn; do
	bash train.sh $data_train $targ_emb $hs 0 0
	bash train.sh $data_train $targ_emb $hs 1 1
	bash train.sh $data_train $targ_emb $hs 1 0
done