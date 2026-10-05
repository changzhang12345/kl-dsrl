# KL-DSRL 接力记录

最后更新：2026-10-05

## 研究问题

在 robomimic Square 上做 DSRL（diffusion steering RL）微调，并给噪声策略加 KL(π_W(·|s) ‖ N(0, I)) 惩罚，系数为 β。目标是看 β 对最终成功率的影响。

- base policy：DPPO 的低维 diffusion MLP，训练时冻结不动，见 `cfg/robomimic/kl_dsrl_square.yaml` 的 `base_policy_path`
- 算法：`stable-baselines3/stable_baselines3/dsrl/kl_dsrl.py`（继承 `DSRL`，在 actor loss 里加 `kl_coef * KL`）
- 训练入口：`train_kl_dsrl.py`，配置：`cfg/robomimic/kl_dsrl_square.yaml`

## 当前进行中

**Slurm 作业 28348710**（2026-10-05 提交，脚本 `run_kl_dsrl_500k.sh`）

- 5 个 β × 3 个 seed = 15 个任务，全部从头训到 500k
- 编号 i：β = (0, 0.1, 0.5, 1, 2)[i % 5]，seed = (1, 2, 3)[i / 5]
- 每个任务 1 张 GPU、8 个 CPU，上限 15 小时；同一时间最多跑 5 个（`--array=0-14%5`）
- 预计每个任务 4 到 5 小时，全部跑完约 13 到 15 小时（不含排队）
- 输出：`logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta{0,0p1,0p5,1,2}_seed{1,2,3}_500000/<时间戳>/checkpoint/`
  - 每 40k 步存一次 `ft_policy_<step>_steps.zip`，结束时存 `final.zip`
  - 结束时额外存 `final_replay_buffer.pkl`（约 0.6GB），以后可以带完整 buffer 续训
- 日志：`logs/slurm/kl_dsrl_square_500k_28348710_<i>.{out,err}`

查看进度：

```bash
squeue -u cz493
grep time_elapsed logs/slurm/kl_dsrl_square_500k_28348710_0.out | tail -1
grep total_timesteps logs/slurm/kl_dsrl_square_500k_28348710_0.out | tail -1
```

之前的速度大约是每步 0.026 秒（250k 步用了 6626 秒）。如果第一批明显更慢，要重新评估 15 小时够不够。

## 已有结果（250k，seed 1，每个点 500 次 rollout）

评估脚本是 `run_eval_kl_dsrl_500ep.sh`（作业 28137209，eval seed 101），结果在 `eval/kl_dsrl_sweep_500ep/beta*/kl_dsrl_checkpoint_eval.csv`。

| step | β=0 | β=0.1 | β=0.5 | β=1 | β=2 |
|---|---|---|---|---|---|
| 40k | 0.526 | 0.428 | 0.726 | 0.614 | 0.614 |
| 80k | 0.698 | 0.654 | 0.688 | 0.662 | 0.658 |
| 120k | 0.602 | 0.576 | 0.682 | 0.700 | 0.668 |
| 160k | 0.670 | 0.696 | 0.750 | 0.804 | 0.792 |
| 200k | 0.736 | 0.650 | 0.798 | 0.840 | 0.780 |
| 240k | 0.776 | 0.746 | 0.846 | 0.818 | 0.812 |
| 250k | 0.784 | 0.818 | **0.886** | 0.844 | 0.846 |

结论：
- β=0.5 最好。它比 β=0 高约 10 个点（约 4σ），这个差距是实的。
- β=0.5 和 β=1、β=2 差约 4 个点，只有约 2σ，单个 seed 还说明不了问题。
- 到 250k 时所有 β 都还在涨，看起来没收敛，所以才重跑 500k。

图：`eval/plots/kl_dsrl_500ep_success.png`（脚本 `plot_kl_dsrl_500ep.py`）。左图是成功率随训练步数的曲线，右图是最终 checkpoint 的 95% 置信区间。

## 这轮新增的代码（已 push 到 `kl-dsrl` 远程）

- `train_kl_dsrl.py`
  - `+resume_path=<ckpt.zip>`：从 checkpoint 接着训练，`num_timesteps` 不清零，`total_timesteps` 填最终的总步数
  - `+resume_replay_buffer=<buffer.pkl>`：续训时加载保存好的 buffer。不给的话会先用当前策略收集 `resume_warmup_steps` 步（默认等于 `init_rollout_steps`）来填 buffer
  - `+save_final_replay_buffer=True`：训练结束时保存 replay buffer
- `utils.py`：`collect_policy_rollouts`，续训时用当前策略填 buffer
- `run_kl_dsrl_500k.sh`：当前这次 15 个任务的扫参
- `run_resume_kl_dsrl.sh`：从旧的 250k checkpoint 续训到 500k（**已废弃**，改成从头训了，留作参考）
- `plot_kl_dsrl_500ep.py`：上面那张图

续训功能只在 CPU 上做过短测试（250000 → 250080 步，能正常加载和保存）。保存和加载 buffer 那两段没实测过。

## 已知坑

- **旧的 250k run 没存 replay buffer**（当时 `save_replay_buffer=False`），只能有损续训，所以这次选择从头训。
- **critic_noise 的优化器状态不在 checkpoint 里**（见 `DSRL._get_torch_save_params`），续训时它的 Adam 状态会重置。
- **评估脚本里的 run 路径是写死的**：`eval_all_kl_dsrl_checkpoints.py:41-45` 只指向 seed1 的 250k run。评估新 run 前要改成新目录（`..._seed{S}_500000`），并加上 `total_timesteps=500000`（`final.zip` 的步数从这里取）。
- **CSV 里两个步数列的区别**：`total_timesteps` = `checkpoint_step` × `n_envs`（4）。`checkpoint_step` 就是 SB3 的 `num_timesteps`，已经包含 4 个并行环境。我的图用的是 `checkpoint_step`。
- **git 远程**：只往 `kl-dsrl`（changzhang12345/kl-dsrl）推。`origin` 是上游 ajwagen/dsrl，不要推。工作区里还有之前没提交的改动（cfg、env_utils、kl_dsrl.py 等），不是这轮加的。

## 下一步

1. 等作业 28348710 跑完，检查 15 个任务是不是都正常结束（`.err` 里没报错，每个 run 都有 `final.zip`）。
2. 修改评估脚本的 run 路径，让它支持多个 seed，然后对所有 checkpoint 各做 500 次 rollout。可以参考 `run_eval_kl_dsrl_500ep.sh`，改成 β × seed 的作业数组。
3. 画每个 β 的 3-seed 平均曲线和置信区间，回答两个问题：(a) 500k 时收敛了没有；(b) β=0.5 和 β=1、β=2 之间有没有显著差异。
4. 视觉输入暂时不做。DPPO 里有图像版的预训练配置 `dppo/cfg/robomimic/pretrain/square/pre_diffusion_mlp_img.yaml`，但图像数据和训练好的权重都还没有。等低维的结论稳定后，再看要不要作为扩展实验。
