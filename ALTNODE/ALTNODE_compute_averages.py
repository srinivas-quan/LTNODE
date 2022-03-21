import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import argparse
import os
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion"],)
parser.add_argument('--savedir', default='./results/', type=str,
                    help='path where to save checkpoints (default: ./results/)')
args = parser.parse_args()
model = 'simple'
dataset = args.dataset

Savedir = args.savedir
rot = 45
attack = 'FGSM'

Entropy_count = []
Confidence_count = []
Error = []
Accuracy = []
Accuracy_props = []
Prob_vec = []
target_entropy_count = []
target_confidence_count = []
err_props = []
conf_count = []
acc_count = []
results = []
miss = []
ood = []
sour_entropy = []
tar_entropy = []

Range = [0,1,2,3,4]
for ID in Range:
	
	savedir = Savedir
	savedir += "/" + "_".join([dataset, model, "DUN", f"warm{0}"]) 
	savedir += "_wd"
	savedir += f"_{ID}.0"
	
	#######Robustness##################
	Entropy_count.append(np.load(savedir+'/_'+attack+'_entropy_count_ATUNODE.npy'))
	Confidence_count.append( np.load(savedir+'/_'+attack+'_confidence_count_ATUNODE.npy'))
	Error.append( np.load(savedir+'/_'+attack+'_Error_ATUNODE.npy'))
	Accuracy.append(1.0 - np.load(savedir+'/_'+attack+'_Error_ATUNODE.npy'))
	#Prob_vec .append( np.load(savedir+'/_'+attack+'_Prob_vec_ATUNODE.npy')
	
	#########OOD###############3
	target_entropy_count.append( np.load(savedir+'/_ood_entropy_count_ATUNODE.npy'))
	target_confidence_count.append( np.load(savedir+'/_ood_confidence_count_ATUNODE.npy'))
	err_props.append( np.load(savedir+'/_err_props_ATUNODE.npy') )
	Accuracy_props.append(1.0 - np.load(savedir+'/_err_props_ATUNODE.npy') )
	miss.append(np.load(savedir+'/mis.npy'))
	ood.append(np.load(savedir+'/OOD.npy'))
	sour_entropy.append(np.load(savedir+'/_entropy_source_target.npy')[0])
	tar_entropy.append(np.load(savedir+'/_entropy_source_target.npy')[1])

	#########rotation   #####333
	conf_count.append( np.load(savedir+'/_conf_count_ATUNODE%d.npy'%(rot)))#rejection plot
	acc_count.append( np.load(savedir+'/_acc_count_ATUNODE%d.npy'%(rot)))
	results.append( np.load(savedir+'/_results_ATUNODE.npy'))

def compute_mean_and_std(value):
	return np.mean(value,0),np.std(value,0)

Entropy_count = np.array(Entropy_count)
Confidence_count = np.array(Confidence_count)
Error = np.array(Error)
Accuracy = np.array(Accuracy)

target_entropy_count = np.array(target_entropy_count)
target_confidence_count = np.array(target_confidence_count)
err_props = np.array(err_props)
Accuracy_props = np.array(Accuracy_props)
OOD = np.array(ood)
Sour_entropy = np.array(sour_entropy)
Tar_entropy = np.array(tar_entropy)

conf_count = np.array(conf_count)
acc_count = np.array(acc_count)
results = np.array(results)


print(Entropy_count.shape)
print(Confidence_count.shape)
print(Error.shape)

print(target_entropy_count.shape)
print(target_confidence_count.shape)
print(err_props.shape)

print(conf_count.shape)
print(acc_count.shape)
print(results.shape)


Entropy_count_ms = compute_mean_and_std(Entropy_count)
Confidence_count_ms = compute_mean_and_std(Confidence_count)
Error_ms  = compute_mean_and_std(Error)
Accuracy_ms  = compute_mean_and_std(Accuracy)


target_entropy_count_ms = compute_mean_and_std(target_entropy_count)
target_confidence_count_ms = compute_mean_and_std(target_confidence_count)
err_props_ms = compute_mean_and_std(err_props)
Accuracy_props_ms = compute_mean_and_std(Accuracy_props)

conf_count_ms = compute_mean_and_std(conf_count)
acc_count_ms = compute_mean_and_std(acc_count)
results_ms = compute_mean_and_std(results)


np.save(Savedir+'/_'+dataset+'_'+attack+'_entropy_count_ATUNODE.npy',Entropy_count_ms)
np.save(Savedir+'/_'+dataset+'_'+attack+'_confidence_count_ATUNODE.npy',Confidence_count_ms)
np.save(Savedir+'/_'+dataset+'_'+attack+'_Error_ATUNODE.npy',Error_ms)
#np.save(savedir+'/_'+attack+'_Prob_vec_TUNODE.npy',Prob_vec)
#print(Error_ms)


np.save(Savedir+'/'+dataset+'_ood_entropy_count_ATUNODE.npy',target_entropy_count_ms)
np.save(Savedir+'/'+dataset+'_ood_confidence_count_ATUNODE.npy',target_confidence_count_ms)
np.save(Savedir+'/'+dataset+'_err_props_ATUNODE.npy',err_props_ms)
np.save(Savedir+'/'+dataset+'_acc_props_ATUNODE.npy',Accuracy_props_ms)


np.save(Savedir+'/'+dataset+'_conf_count_ATUNODE%d.npy'%(rot),conf_count_ms)
np.save(Savedir+'/'+dataset+'_acc_count_ATUNODE%d.npy'%(rot),acc_count_ms)
np.save(Savedir+'/'+dataset+'_results_ATUNODE.npy',results_ms)

print("source entropy",compute_mean_and_std(Sour_entropy))
print("Tar entropy",compute_mean_and_std(Tar_entropy))
print('OOD ROC',compute_mean_and_std(OOD))
print("Error\n",Error_ms[0][0],Error_ms[1][0])

style = 'seaborn-whitegrid'
fig, ax = plt.subplots()
plt.style.use(style)
print(err_props_ms[0].shape)
ax.plot(np.arange(0, 1, 0.05), Accuracy_props_ms[0][0::10],'o--',color='brown',label='NODE')
plt.fill_between(np.arange(0, 1, 0.05),Accuracy_props_ms[0][0::10]-Accuracy_props_ms[1][0::10] , Accuracy_props_ms[1][0::10]+Accuracy_props_ms[0][0::10],color='#888888', alpha=0.4)
plt.savefig(Savedir+'/plot_test.pdf')