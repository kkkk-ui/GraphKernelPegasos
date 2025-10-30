import matplotlib.pyplot as plt
import numpy as np

# lambda 値
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10, 100]

# 枝刈りなしの結果
time_no_prune = [9.4188, 19.0890, 43.0868, 80.8836, 201.6332, 239.7608, 258.5186, 256.3522]
acc_no_prune = [0.9981, 0.9981, 0.9943, 0.9338, 0.7656, 0.6446, 0.6087, 0.6276]
base_no_prune = [18, 28, 68, 171, 400, 478, 472, 472]

# 枝刈りありの結果
time_prune = [11.5602, 17.9117, 21.5783, 50.2306, 136.4084, 147.7677, 172.1772, 180.5005]
acc_prune = [0.9887, 0.8979, 0.8639, 0.9187, 0.8034, 0.6049, 0.6427, 0.6314]
base_prune = [16, 28, 34, 89, 254, 253, 275, 271]

time_nc_prune = [15.7494, 12.2922, 14.0270, 33.4471, 73.4886, 87.7436, 98.4539, 87.0964]
acc_nc_prune  = [0.9281663516068053, 0.9111531190926276, 0.9376181474480151, 0.9376181474480151,
               0.7901701323251418, 0.6351606805293005, 0.610586011342155, 0.6294896030245747]
base_nc_prune = [31, 27, 36, 92, 231, 265, 271, 260]


# cython-枝刈りありの結果
time_c_prune = [4.6966, 10.6492, 12.3738, 33.1161, 74.1134, 98.5616, 91.7693, 108.0269]
acc_c_prune = [0.9981, 0.9660, 0.7921, 0.9244, 0.7089, 0.6389, 0.6238, 0.6011]
base_c_prune = [12, 25, 36, 81, 243, 287, 274, 270]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 1. 処理時間
axs[0].plot(lambdas, time_no_prune, marker='o', label='No Pruning', linestyle='--')
axs[0].plot(lambdas, time_prune, marker='s', label='Pruning')
axs[0].plot(lambdas, time_c_prune, marker='s', label='cython Pruning', linestyle=':')
axs[0].plot(lambdas, time_nc_prune, marker='o', label='not cython Pruning', linestyle='-')
axs[0].set_xscale("log")
axs[0].set_ylabel("Time (sec)")
axs[0].set_title("Training Time")
axs[0].set_xticks(lambdas)
axs[0].legend()
axs[0].grid(True)

# 2. 精度
axs[1].plot(lambdas, acc_no_prune, marker='o', label='No Pruning', linestyle='--')
axs[1].plot(lambdas, acc_prune, marker='s', label='Pruning')
axs[1].plot(lambdas, acc_c_prune, marker='s', label='cython Pruning', linestyle=':')
axs[1].plot(lambdas, acc_nc_prune, marker='o', label='not cython Pruning', linestyle='-')
axs[1].set_xscale("log")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Accuracy")
axs[1].set_xticks(lambdas)
axs[1].legend()
axs[1].grid(True)

# 3. 基底数
axs[2].plot(lambdas, base_no_prune, marker='o', label='No Pruning', linestyle='--')
axs[2].plot(lambdas, base_prune, marker='s', label='Pruning')
axs[2].plot(lambdas, base_c_prune, marker='s', label='cython Pruning', linestyle=':')
axs[2].plot(lambdas, base_nc_prune, marker='o', label='not cython Pruning', linestyle='-')
axs[2].set_xscale("log")
axs[2].set_ylabel("Support Vectors")
axs[2].set_xlabel("Lambda")
axs[2].set_title("Number of Basis (Support Vectors)")
axs[2].set_xticks(lambdas)
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
