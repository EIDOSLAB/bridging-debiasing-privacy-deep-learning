import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm

df_vanilla = pd.read_csv('./imdb_features_EB1_vanilla_age.csv').rename(columns={'age': 'gender'})
df_end = pd.read_csv('./imdb_features_EB1_end_age.csv').rename(columns={'age': 'gender'})

print(df_vanilla.head())

def get_mean_std(df):
    pca = PCA(n_components=1)
    data = pca.fit_transform(df.iloc[:, 2:])
    data = np.array(data[:,0])

    idx_male = df.gender == 0
    idx_female = df.gender == 1

    male = data[idx_male]
    female = data[idx_female]

    assert len(male)+len(female) == len(data)

    mean_m, std_m = norm.fit(male)
    mean_f, std_f = norm.fit(female)

    return [mean_m, mean_f], [std_m, std_f]


SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

from matplotlib import rc
rc('axes', titlesize=18)     # fontsize of the axes title
rc('axes', labelsize=18)    # fontsize of the x and y labels
rc('xtick', labelsize=14)    # fontsize of the tick labels
rc('ytick', labelsize=14)    # fontsize of the tick labels
rc('legend', fontsize=14)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
rc('font', size=18)
#rc('font', family='Times New Roman')
rc('text', usetex=True)

scale = 0.8
x_left, x_right = -3, 3
x_axis = np.arange(x_left, x_right, 0.001)

mean, std = get_mean_std(df_vanilla)
print('mean', mean)
print('std', std)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
y = norm.pdf(x_axis, mean[0], std[0])*0.8
m1 = y.max()
ax.plot(x_axis, y, label='Male')
ax.fill_between(x_axis, y1=y, y2=y-y, alpha=0.1)
ax.vlines(mean[0], ymin=0, ymax=y.max(), linestyle='--', color='gray')

y = norm.pdf(x_axis, mean[1], std[1])*0.8
m2 = y.max()
ax.plot(x_axis, y, label='Female')
ax.fill_between(x_axis, y1=y, y2=y-y, alpha=0.1)
ax.vlines(mean[1], ymin=0, ymax=y.max(), linestyle='--', color='gray')

ax.legend()
ax.set_xlabel('PC', loc='right')
#ax.set_xticks([mean[0], mean[1]])
#ax.set_xticklabels([r'$\mu_m$', r'$\mu_f$'])
ax.set_xlim(x_left, x_right)
plt.text(mean[0]-0.1, -0.1, r'$\mu_m$')
plt.text(mean[1]-0.1, -0.1, r'$\mu_f$')
ax.set_yticks([])
ax.set_ylim(0, max(m1, m2) + 0.2)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.tick_params(axis=u'both', which=u'both', length=0)

plt.savefig('normal_vanilla.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

mean, std = get_mean_std(df_end)
print('mean', mean)
print('std', std)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
y = norm.pdf(x_axis, mean[0], std[0])*0.8
m1 = y.max()
ax.plot(x_axis, y, label='Male')
ax.fill_between(x_axis, y1=y, y2=y-y, alpha=0.1)
ax.vlines(mean[0], ymin=0, ymax=y.max(), linestyle='--', color='gray')

y = norm.pdf(x_axis, mean[1], std[1])*0.8
m2 = y.max()
ax.plot(x_axis, y, label='Female')
ax.fill_between(x_axis, y1=y, y2=y-y, alpha=0.1)
ax.vlines(mean[1], ymin=0, ymax=y.max(), linestyle='--', color='gray')

ax.legend()
ax.set_xlabel('PC', loc='right')
#ax.set_xticks([mean[0]+0.1, mean[1]-0.1])
#ax.set_xticklabels([r'$\mu_m$', r'$\mu_f$'])
plt.text(mean[0], -0.05, r'$\mu_m$')
plt.text(mean[1]-0.2, -0.05, r'$\mu_f$')
ax.set_xlim(x_left, x_right)

ax.set_yticks([])
ax.set_ylim(0, max(m1, m2) + 0.2)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.tick_params(axis=u'both', which=u'both', length=0)

plt.savefig('normal_end.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
