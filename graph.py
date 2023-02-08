import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time

def data_process(data, size):
	means = []
	x = []
	stds = []
	for i in range(len(data)):
		means.append(np.mean(data[max(i - size, 0):i + 1]))
		x.append(i)
		stds.append(np.std(data[max(i - size, 0):i + 1]))
	stds = np.array(stds)

	return x, means, stds

def data_fill(data):
	temp = np.array(data[-20:])
	while len(data) < 1000:
		data += temp[np.random.randint(20, size=20)].tolist()
	return data[:1000]

env = 'Hopper-v3'
# env = 'Walker2d-v3'
seed = 0

plt.title(env,fontsize=20)
if os.path.isfile("./figures/returns_"+env+'(seed:'+str(seed)+").pickle"):
	with open("./figures/returns_"+env+'(seed:'+str(seed)+").pickle","rb") as fr:
		data_t = pickle.load(fr)
		# data0 = data_fill(data_t)
		data0=data_t
		data = np.array(data0)
# print(data)
# number=1
# data0 = data.mean(0)
# x0, means0, stds0 = data_process(data0.tolist(), number)
# data_M = data.max(0)
# x_M, means_M, stds_M = data_process(data_M.tolist(), number)
# data_m = data.min(0)
# x_m, means_m, stds_m = data_process(data_m.tolist(), number)
x_m, means_m, stds_m = data_process(data.tolist(), 1)
plt.plot(means_m)
# plt.fill_between(x0, means_m, means_M, alpha=0.25)

ax = plt.gca()
ax.set_facecolor('#E7EAEF')
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tick_params(top=False, bottom=False, left=False, right=False)
plt.grid(True,color='white',alpha=1)
plt.xlabel('steps',fontsize=15)
plt.ylabel('average return',fontsize=15)
plt.legend(loc='upper left',fontsize=13)
plt.subplots_adjust(left=0.12, right=0.97, top=0.935, bottom=0.110)

# plt.legend(loc='lower right')
plt.savefig(env+'.png',dpi=600)
plt.close()
# plt.show()
