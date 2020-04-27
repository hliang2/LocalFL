python3 NLocalSGD.py\
       --lr 0.01 --bs 10 --cr 100 --cp 20 --slowRatio 0.5\
       --alpha 2 --beta 2 --globalCp 40\
       --model logistic\
       --save -p --name test2\
       --size 10 --total_size 50 --backend nccl --NIID --Unbalanced  

python3 NLocalSGD.py\
       --lr 0.01 --bs 10 --cr 100 --cp 20\
       --alpha 2 --beta 2 --mu 0.5 --globalCp 40\
       --model logistic\
       --save -p --name test4\
       --size 10 --total_size 50 --backend nccl --NIID --Unbalanced --FedProx --persistent     