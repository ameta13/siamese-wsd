siamese=true

for data_train in wic_train-en-en wic; do
	bash train.sh $data_train comb_admn -1 0 0 $siamese
	bash train.sh $data_train dist_l1ndotn 0 1 1 $siamese
done