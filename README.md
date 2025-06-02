# GraphKernelPegasos

---

## 使用データセット（データ出典）
本プロジェクトでは、Twitterボット検出のためのグラフデータセット TwiBot-22 の c-15 を使用しています。
このデータセットは以下の論文に基づいて提供されています：

@inproceedings{fengtwibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and others},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}

TwiBot-22は、ユーザー間のフォロー関係や投稿内容をグラフ構造として表現した大規模データセットであり、グラフカーネルを用いたボット検出のベンチマークとして活用されています。

## インストール方法

まずはリポジトリをクローンしてください：

```bash
git clone https://github.com/kkkk-ui/GraphKernelPegasos.git
cd GraphKernelPegasos
