import matplotlib.pyplot as plt
import numpy as np

# acc は小数点以下4桁に丸め済み
t = [20, 50, 100, 300, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000]
acc = [0.7788, 0.8847, 1.0, 0.9622, 0.9792, 0.9282, 0.9924, 0.9754, 0.9301, 0.9905, 0.9452, 0.9017]
time_sec = [0.4602, 2.5826, 2.8155, 13.2649, 41.5285, 220.6341, 742.9481, 1552.1726, 2235.6162, 2975.7356, 8967.5523, 25713.8633]
base = [5, 5, 7, 48, 92, 163, 230, 277, 309, 327, 395, 393]

t_dg = [20, 50, 100, 300, 500, 1000, 2000, 3000, 4000]
time_dg = [0.2237, 0.7131, 2.5059, 13.5724, 39.5327, 103.8466, 343.0696, 760.6545, 1203.0420]
acc_dg = [0.9112, 0.9773, 0.8355, 0.9546, 0.8696, 0.9698, 0.9698, 0.9471, 0.9773]
base_dg  = [6, 20, 39, 67, 88, 150, 249, 334, 382]


# ========== 1) Training Time ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, time_mean, yerr=time_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, time_sec, '-', marker='o', label='Sparse Dict Pruning')
plt.plot(t_dg, time_dg, '-', marker='o', label='Sparse Dict Growing')
# plt.xscale("log")
plt.ylabel("Time [s]")
plt.xlabel("iter")
plt.title("Training Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_per_iter.png", dpi=600)
plt.show()

# ========== 2) Accuracy ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, acc_mean, yerr=acc_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, acc, '-', marker='o', label='Sparse Dict Pruning')
plt.plot(t_dg, acc_dg, '-', marker='o', label='Sparse Dict Growing')
# plt.xscale("log")
plt.ylabel("Accuracy")
plt.xlabel("iter")
plt.title("Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("acc_per_iter.png", dpi=600)
plt.show()

# ========== 3) Number of Basis ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, base_mean, yerr=base_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, base, '-', marker='o', label='Sparse Dict Pruning')
plt.plot(t_dg, base_dg, '-', marker='o', label='Sparse Dict Growing')
# plt.xscale("log")
plt.ylabel("Support Vectors")
plt.xlabel("iter")
plt.title("Number of Basis (Support Vectors)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("base_per_iter.png", dpi=600)
plt.show()
