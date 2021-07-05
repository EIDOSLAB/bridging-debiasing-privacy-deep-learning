import pandas as pd
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#sns.set()
#sns.color_palette("mako")

df = pd.read_csv('bias_classifier.csv')
print(df.head())

print(df.groupby(['crit', 'rho']).count())

crits = df.crit.unique()
rhos = df.rho.unique()

accuracy_bias = {}
std_bias = {}

accuracy_target = {}
std_target = {}

for crit in crits:
    for rho in rhos:
        print('\n', crit, rho)
        acc_bias = df.loc[(df.crit == crit) & (df.rho == rho)]['unbiased.accuracy.bias'].values
        acc_target = df.loc[(df.crit == crit) & (df.rho == rho)]['unbiased.accuracy.f_target'].values
        
        print('WHAT', acc_bias, acc_target)

        if crit not in accuracy_bias:
            accuracy_bias[crit] = []
            accuracy_target[crit] = []
            std_bias[crit] = []
            std_target[crit] = []
        
        accuracy_bias[crit].append(np.mean(acc_bias)*100.)
        accuracy_target[crit].append(np.mean(acc_target)*100.)
        std_bias[crit].append(np.std(acc_bias)*100.)
        std_target[crit].append(np.std(acc_target)*100.)

      
plt.figure()
for crit in crits:
    mean, std = np.array(accuracy_bias[crit]), np.array(std_bias[crit])
    plt.plot(mean, label=crit, marker='o')
    plt.fill_between(x=range(len(mean)), y1=mean-std, y2=mean+std, alpha=0.25)
    #plt.errorbar(range(len(mean)), mean, yerr=std)

plt.xticks(range(len(rhos)), rhos)
plt.legend()
plt.xlabel('rho')
plt.ylabel('accuracy')
plt.savefig('bias_acc.pdf', format='pdf', dpi=300)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)

for crit in crits[::-1]:
    ax.scatter(accuracy_bias[crit], accuracy_target[crit], alpha=0.5, label=crit)
#plt.xticks(range(len(rhos)), rhos)
ax.legend()
ax.text(1, 100, 'Fully Unbiased')
ax.text(100, 1, 'Fully Biased')
ax.set_xlabel('Bias Accuracy', loc='right')
ax.set_ylabel('Target Accuracy', loc='bottom')
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')


plt.savefig('scatter.pdf', format='pdf', dpi=300)
plt.show()

