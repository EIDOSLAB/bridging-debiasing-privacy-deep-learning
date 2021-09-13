import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

#sns.set()
#sns.color_palette("mako")

df = pd.read_csv('bias_classifier.csv')
df = df.replace({'vanilla': 'Vanilla', 'end': 'EnD', 'rebias': 'ReBias', 'rubi': 'RUBi', 'learned-mixin': 'LearnedMixin'})
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
        
        if crit not in accuracy_bias:
            accuracy_bias[crit] = []
            accuracy_target[crit] = []
            std_bias[crit] = []
            std_target[crit] = []
        
        accuracy_bias[crit].append(np.mean(acc_bias)*100.)
        accuracy_target[crit].append(np.mean(acc_target)*100.)
        std_bias[crit].append(np.std(acc_bias)*100.)
        std_target[crit].append(np.std(acc_target)*100.)


SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

from matplotlib import rc
rc('axes', titlesize=18)     # fontsize of the axes title
rc('axes', labelsize=18)    # fontsize of the x and y labels
rc('xtick', labelsize=18)    # fontsize of the tick labels
rc('ytick', labelsize=18)    # fontsize of the tick labels
rc('legend', fontsize=14)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
rc('font', size=18)
#rc('font', family='Times New Roman')
rc('text', usetex=True)

#plt.rcParams['font.family'] = 'Times New Roman' 
#plt.rcParams.update({ 
#    'font.size': 12, 
#    'text.usetex': True, 
#    'text.latex.preamble': r'\usepackage{amsfonts}' 
#})

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for crit in crits:
#     mean, std = np.array(accuracy_bias[crit]), np.array(std_bias[crit])
#     ax.plot(mean[::-1], label=crit, marker='o', linestyle='-.',)
#     ax.fill_between(x=range(len(mean)), y1=(mean-std)[::-1], y2=(mean+std)[::-1], alpha=0.25)

# plt.xticks(range(len(rhos)), rhos[::-1])
# ax.legend()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.text(-0.125, -5, 'High Bias', fontstyle='italic')
# ax.text(2.75, -5, 'Low Bias', fontstyle='italic')
# ax.set_xlabel(r'$\rho$')
# ax.set_ylabel('Bias Accuracy')
# plt.savefig('bias_acc.pdf', format='pdf', dpi=300)
# plt.show()

crits = ['RUBi', 'Vanilla', 'ReBias', 'LearnedMixin', 'EnD']
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')

angle_offset = 15
angle_width = 50

for i, crit in enumerate(crits):
    color=next(ax._get_lines.prop_cycler)['color']
    mean, std = np.array(accuracy_bias[crit]), np.array(std_bias[crit])
    
    x = rhos[::-1]
    y = mean[::-1]
    
    x = np.radians([angle_offset, angle_offset+angle_width, angle_offset+angle_width*2, angle_offset+angle_width*3])
    ax.scatter(x, y, label=crit, c=color, alpha=0.5)
    
    f = interp1d(x, y, kind='quadratic')
    xnew = np.linspace(np.radians(angle_offset), np.radians(angle_offset+angle_width*3))
    ax.plot(xnew, f(xnew), c=color, linestyle='--')
    ax.fill_between(x=xnew, y1=f(xnew), y2=f(xnew)-f(xnew), alpha=0.05) #1./float(len(crits)-(i-0.8)))

#color=next(ax._get_lines.prop_cycler)['color']
#mean, std = np.array([10.]*4), np.array([0.]*4)
#x = range(len(rhos))
#y = mean[::-1]
#xnew = np.linspace(0, len(rhos)-1)
#ax.plot(xnew, [10.]*len(xnew), c='gray', linestyle='--', label='Unbiased')

ax.legend(bbox_to_anchor=(1.0,1.15), markerscale=2)

thetaticks = np.arange(angle_offset, angle_offset+200, 50)
ax.set_thetagrids(thetaticks, labels=rhos[::-1])

ax.tick_params(axis='x', pad=15)
ax.set_ylim(0, 110)
ax.set_thetamin(angle_offset)
ax.set_thetamax(angle_offset+angle_width*3)

ax.spines['polar'].set_color('none')
ax.yaxis.set_label_position('right')
ax.xaxis.set_label_position('bottom')

ax.text(np.radians(-50), ax.get_rmax()/5, 'Private Class Accuracy', rotation=15)
ax.text(np.radians(60), ax.get_rmax(), r'$\rho$')

plt.tight_layout()
plt.savefig('bias_acc_polar.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

sizes = np.array([1, 2.5, 5.,10])*40
for crit in crits:
    print(rhos)
    print(accuracy_bias[crit])
    ax.scatter(accuracy_bias[crit], accuracy_target[crit], alpha=0.5, label=crit, s=sizes)
#plt.xticks(range(len(rhos)), rhos)
ax.plot(1, 50, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(50, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
ax.legend(markerscale=1)
ax.text(1, 100, 'Privacy preserving', fontstyle='italic')
ax.text(90, 1, 'Privacy leakage', fontstyle='italic')
ax.set_xlabel('Private Class Accuracy', loc='right')
ax.set_ylabel('Target Accuracy', loc='bottom', labelpad=-60)
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')

#plt.tight_layout()
plt.savefig('scatter.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

print(accuracy_bias, std_bias)

crits = ['Vanilla', 'RUBi', 'ReBias', 'LearnedMixin', 'EnD']
table = []
for crit in crits:
    entry = [crit]
    accs = accuracy_bias[crit]
    accs = [f'{a:.2f}' for a in accs]
    entry.extend(accs)
    table.append(entry)

table = tabulate(table, headers=['Method', '0.99', '0.995', '0.997', '0.999'], tablefmt='latex_booktabs')
print(table)

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
