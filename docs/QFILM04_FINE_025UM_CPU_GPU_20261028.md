# Qfilm0.4：0.25 μm 像面采样与 11 处方 CPU/GPU 完整任务

本轮开始于北京时间 2026-10-28。**代码和解析 Airy 控制已完成；11 镜头完整 CPU/GPU PSF/MTF 尚未取得结果。** 任务提交、前检与计时不能作为完整计算或精度通过证据。

## 改动与固定条件

- Qfilm0.4 API 默认配置及 CLI 默认像面步距改为 **0.25 μm**。
- 保留原来 **0.64 mm × 0.64 mm** 支持窗口，因此像面网格使用 **2560²**，不再使用 128²。
- 256² 瞳面、float64、真实物理出射平面、直接 Huygens、原有 Jacobian 振幅和复 OTF 多波长合成保持一致。
- Qfilm0.3 的旧默认值保留。显式传入旧物理配置仍可复现历史测试。
- CPU/GPU 完整任务采用 36×24 mm、径向像高 0 / 10.81665 / 21.63331 mm，430/470/510/550/590/630/670 nm，5–80 lp/mm 的 T/S MTF。每种后端应尝试 **11×3×7 = 231 张 PSF**。
- 七波长权重仍为 0.075/0.125/0.16/0.28/0.16/0.125/0.075。固定机械像面，不逐色调焦。

像素数增加 400 倍；直接积分的计算量也约增加 400 倍。不能通过缩小窗口或降低瞳面采样宣称完成同一测试。

## 案例与来源

固定 DeepLens 源码版本 `911bb9d334fce7e20a4407c4506ac683b52ff246`。
11 个文件为 `ef100mm_f2.8`、`ef35mm_f2.0`、`ef40mm_f2.8`、`ef50mm_f1.8`、`ef85mm_f1.8`、`rf16mm_f2.8`、`rf24mm_f1.8`、`rf35mm_f1.8`、`rf50mm_f1.8`、`sigma70mm_f2.8`、`yongnuo_50mm_f1.8`。
多数为商品关联公开专利处方，不能视为工厂量产处方或实物标定。源文件逐例 SHA256 与来源说明随结果保存。

## 已实际得到的证据

1. 8 组 Airy 解析控制重跑完成。0.25 μm 的最大绝对 MTF 偏差为 **0.000444668–0.002089192**。这些是解析控制的有限窗口结果，不是摄影镜头误差。
2. CPU 上 11 个处方均完成中心 550 nm 的 16² 像素计时小块，使用真实 256² 瞳面。这些仅用于测量成本，**不是完整 PSF，也没有最终 J**。
3. 默认网格、0.64 mm 支持、无效步距拒绝及新增进度回调的回归检查通过。回调启用/禁用的测试 PSF 逐元素相同；该回归使用小合成案例，不是实物验证。
4. 当前执行环境为 PyTorch 2.15.0+cpu，没有 CUDA。GPU 入口已实际尝试并明确返回 `unavailable`，没有回退为 CPU。
5. 工作站最后读取到的心跳为北京时间 **2026-10-28 20:25:48**，未见恢复或本轮任务运行证据。这个时间戳不能代表当前 GPU 空闲状态。

11 个处方的完整瞳面几何前检已结束：231 组中 196 组同时具有足够透射光线与有效主光线，35 组未通过。EF35 的边缘 7 组主光线无效；RF16/RF24 的半场各 7 组主光线无效、最外场各 7 组没有可用透射光线。这些均不是完整 PSF 或最终评分结果。

`mode=preflight` 或 `mode=benchmark` 的报告均不是正式 PSF 结果。


| 处方 | 中心通过 / 7 | 半场通过 / 7 | 边缘通过 / 7 |
|---|---:|---:|---:|
| ef100mm_f2.8 | 7 | 7 | 7 |
| ef35mm_f2.0 | 7 | 7 | 0 |
| ef40mm_f2.8 | 7 | 7 | 7 |
| ef50mm_f1.8 | 7 | 7 | 7 |
| ef85mm_f1.8 | 7 | 7 | 7 |
| rf16mm_f2.8 | 7 | 0 | 0 |
| rf24mm_f1.8 | 7 | 0 | 0 |
| rf35mm_f1.8 | 7 | 7 | 7 |
| rf50mm_f1.8 | 7 | 7 | 7 |
| sigma70mm_f2.8 | 7 | 7 | 7 |
| yongnuo_50mm_f1.8 | 7 | 7 | 7 |

## 执行与恢复

CPU 例：

```bash
PYTHONPATH=/path/to/DeepLens python validation/qfilm04_fine_11_cpu_gpu.py \
  --backend c --deeplens-root /path/to/DeepLens --threads 8 \
  --output artifacts/fine_cpu.json
```

GPU 将 `--backend c` 改为 `--backend g`；其余光学参数相同。
可通过 `--names ef50mm_f1.8` 独立执行一个案例。`--fields` 可增加径向场点数，默认三个场点；本轮不是 100 场点认证。
使用相同输出路径重跑会校验协议与源文件，并复用已完成的单色 OTF；同一张 PSF 的中途像素进度仅用于监控，不能从中间像素恢复。

每个场点/波长独立记录 `completed` 或 `invalid`，无效场点不会被补成零误差或有效最终评分。
光谱合成采用单色复 OTF 与实际主光线相对位移，再取绝对值。
最终 J 的非 MTF 分量仍使用原有暂定基线，不能解释为这些分量也完成了 0.25 μm 物理认证。

比较命令：

```bash
python validation/qfilm04_compare_fine_cpu_gpu.py \
  --cpu artifacts/fine_cpu.json --gpu artifacts/fine_gpu.json \
  --output artifacts/fine_cpu_gpu_comparison.json
```

比较器校验完整协议及源文件哈希，只比较双方实际完成的 PSF；缺少任一组合即显示 `incomplete`。CPU/GPU 吻合只证明执行设备一致性，不证明独立物理精度。

22 个完整任务已写入 `remote/queue/`，11 个 GPU 在前、11 个 CPU 在后。执行代码固定为 `9a36a7011f379a153e4fab7f8c95b180cece49a1`；队列提交为 `a906fe43fc9a7d89627a92ed73f30f9a9339ce6d`。尚未观察到首个新 GPU 任务被领取。需要恢复工作站 `cots-lamcts-bridge` 服务后才能继续远端执行；当前环境不能代替 CUDA 工作站。

## 结果入口

- [22 个完整任务及执行状态](../artifacts/qfilm04_fine_11_execution_status_20261028.json)
- [11 镜头几何前检摘要 CSV](../artifacts/qfilm04_fine_11_preflight_summary_20261028.csv)
- [0.25 μm Airy 解析控制](../artifacts/qfilm04_airy_sampling_rerun_20261028.json)
- [11 案例 CPU 计时小块](../artifacts/qfilm04_fine_11_cpu_timing_single_thread_20261028.json)
- [完整瞳面 CPU 几何前检](../artifacts/qfilm04_fine_11_cpu_preflight_20261028.json)
- [本地 GPU 可用性检查](../artifacts/qfilm04_fine_11_gpu_availability_20261028.json)
- [网格及回调回归](../artifacts/qfilm04_fine_grid_regression_20261028.json)

`training_eligible=False`、`physics_certified=False`、`sampling_converged=False` 保留。步距变小不能替代窗口、瞳面以及独立参考传播方法的收敛验证。
