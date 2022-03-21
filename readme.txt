The following is developed on the top of https://github.com/cambridge-mlg/DUN 

The code from https://github.com/juntang-zhuang/torch_ACA is used to train our models using Dopri5 numerical method.



Train the models on all the datasets with different seed values(0,1,2,3,4)


1. Install the required packages provided in requirements.txt file 

2. To train ALT-NODE model on MNIST dataset, chnage to ALTNODE direcotry run the following command
  
     python3 train_ALTNODE.py --dataset MNIST --seed 0 --savedir ./results
   
3. ALTNODE models for other datasets CIFAR10 , SVHN, Fashion can be trained by replacing MNIST in the above command with appropriate dataset name.

4. Trained models can be tested for adversarial attack, out-of-distribution experiments, performance under rotation by executing the following, 

		python3 test_ALTNODE.py --seed 0  --dataset MNIST --savedir ./results/
		
5. All the computed results will be saved as .npy files in default direcory ./results, this can be changed by providing the different directory as argument to --savedir	


6. Run python3 LTNODE_compute_averages.py --savedir ./results --dataset MNIST, to compute mean and standard deviation of the performance of all 5 models and are stored in the deafult directory as .npy files.



1. TO train LT-NODE models change to LTNODE directory

2. To train LT-NODE model on MNIST dataset, run the following command
  
    python3 train_LTNODE.py --dataset MNIST --seed 0 --savedir ./results/
   
3. LTNODE models for other datasets CIFAR10 , SVHN, Fashion cab be trained by replacing MNIST in the above command with appropriate dataset name.



4. Trained models can be tested for adversarial attack, out-of-distribution experiments, performance under rotation by executing the following, 

		python3 test_LTNODE.py --seed 0  --dataset MNIST --savedir ./results/
		
5. All the computed results will be saved as .npy files in default directory ./results, this can be changed by providing the different directory as argument to --savedir	

6. Run python3 ALTNODE_compute_averages.py --savedir ./results --dataset MNIST, to compute mean and standard deviation of the performance of all 5 models and are stored in the deafult directory as .npy files.

