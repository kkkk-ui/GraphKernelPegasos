import numpy as np
import matplotlib.pyplot as plt

# ==== 各回のデータ ====
times = np.array([
    [11.4044, 20.6439, 14.1883, 46.6571, 113.9741, 68.4284, 79.7303, 95.9029],
    [10.4991, 19.2867, 6.5233, 65.6547, 56.1565, 77.3743, 73.1427, 61.8679],
    [8.9294, 11.5734, 8.0590, 52.1838, 77.4194, 63.5191, 66.9935, 66.9307],
    [21.1267, 18.0364, 8.1289, 36.4570, 77.1973, 76.4760, 77.4647, 83.0424]
])

accs = np.array([
    [0.7750, 0.9338, 0.9376, 0.9773, 0.8866, 0.9830, 0.9735, 0.9263],
    [0.9641, 0.6257, 0.9206, 0.9679, 0.8677, 0.9943, 0.9603, 0.9168],
    [0.9943, 0.9225, 0.9849, 0.9149, 0.9509, 0.9149, 0.9074, 1.0000],
    [0.7240, 0.8204, 0.8072, 0.9943, 0.8847, 0.9055, 0.9263, 0.9792]
])

bases = np.array([
    [13, 5, 4, 86, 108, 106, 108, 101],
    [9, 28, 5, 104, 98, 124, 116, 107],
    [3, 14, 6, 96, 102, 104, 101, 122],
    [37, 24, 5, 93, 116, 123, 134, 124]
])


# ==== 統計量 ====
time_mean, time_std = times.mean(axis=0), times.std(axis=0)
acc_mean, acc_std = accs.mean(axis=0), accs.std(axis=0)
base_mean, base_std = bases.mean(axis=0), bases.std(axis=0)

lamdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]


# ==== 比較データ ====
time_no_prune = [9.4188, 19.0890, 43.0868, 80.8836, 201.6332, 239.7608, 258.5186, 256.3522]
acc_no_prune = [0.9981, 0.9981, 0.9943, 0.9338, 0.7656, 0.6446, 0.6087, 0.6276]
base_no_prune = [18, 28, 68, 171, 400, 478, 472, 472]

time_nc_prune = [15.7494, 12.2922, 14.0270, 33.4471, 73.4886, 87.7436, 98.4539, 87.0964]
acc_nc_prune  = [0.9282, 0.9112, 0.9376, 0.9376, 0.7902, 0.6352, 0.6106, 0.6295]
base_nc_prune = [31, 27, 36, 92, 231, 265, 271, 260]

time_proj = [315.9307, 109.3227, 2076.8985, 2905.0784, 3229.0533, 2815.9781, 2776.0530, 2877.6462]
acc_proj  = [0.8809073724007561, 0.8657844990548205, 0.9886578449905482, 0.9659735349716446, 0.9149338374291115, 0.9848771266540642, 0.8733459357277883, 0.8998109640831758]
base_proj = [57, 14, 297, 327, 340, 304, 311, 302]


# ==== プロット ====
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# ==== 1. 処理時間 ====
axs[0].errorbar(lamdas, time_mean, yerr=time_std, fmt='-o', capsize=5, color='r', label='Sparse Dict Pruning')
axs[0].plot(lamdas, time_no_prune, '--', marker='s', label='No Pruning')
axs[0].plot(lamdas, time_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
axs[0].plot(lamdas, time_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
axs[0].set_xscale("log")
axs[0].set_ylabel("Time [s]")
axs[0].set_title("Training Time")
axs[0].legend()
axs[0].grid(True)

# ==== 2. 精度 ====
axs[1].errorbar(lamdas, acc_mean, yerr=acc_std, fmt='-o', capsize=5, color='r', label='Sparse Dict Pruning')
axs[1].plot(lamdas, acc_no_prune, '--', marker='s', label='No Pruning')
axs[1].plot(lamdas, acc_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
axs[1].plot(lamdas, acc_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
axs[1].set_xscale("log")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Accuracy Comparison")
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0.5, 1.0)

# ==== 3. 基底数 ====
axs[2].errorbar(lamdas, base_mean, yerr=base_std, fmt='-o', capsize=5, color='r', label='Sparse Dict Pruning')
axs[2].plot(lamdas, base_no_prune, '--', marker='s', label='No Pruning')
axs[2].plot(lamdas, base_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
axs[2].plot(lamdas, base_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
axs[2].set_xscale("log")
axs[2].set_ylabel("Support Vectors")
axs[2].set_xlabel("Lambda")
axs[2].set_title("Number of Basis (Support Vectors)")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
