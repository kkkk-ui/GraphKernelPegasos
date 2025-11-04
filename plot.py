import matplotlib.pyplot as plt
import numpy as np

# acc は小数点以下4桁に丸め済み
t        = [100, 300, 500, 1000, 2000, 3000, 4000]
time_sec = [1.9291, 11.9001, 54.7049, 289.4142, 776.6321, 2046.4504, 2904.4602]
acc      = [0.9376, 0.9754, 0.9603, 0.9282, 0.9698, 0.9698, 0.9830]
base     = [5, 50, 102, 174, 230, 305, 316]

# ========== 1) Training Time ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, time_mean, yerr=time_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, time_sec, '-', marker='o', label='Sparse Dict Pruning')
# plt.xscale("log")
plt.ylabel("Time [s]")
plt.xlabel("iter")
plt.title("Training Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_time.png", dpi=600)
plt.show()

# ========== 2) Accuracy ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, acc_mean, yerr=acc_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, acc, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
# plt.xscale("log")
plt.ylabel("Accuracy")
plt.xlabel("iter")
plt.title("Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("accuracy.png", dpi=600)
plt.show()

# ========== 3) Number of Basis ==========
plt.figure(figsize=(6, 4))
# plt.errorbar(t, base_mean, yerr=base_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(t, base, '-', marker='o', label='Sparse Dict Pruning')
# plt.xscale("log")
plt.ylabel("Support Vectors")
plt.xlabel("iter")
plt.title("Number of Basis (Support Vectors)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("basis_support_vectors.png", dpi=600)
plt.show()
