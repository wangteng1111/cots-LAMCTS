#!/usr/bin/env python3
"""Build a conservative, source-linked report from completed CPU artifacts."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
NAMES={
 "ef100mm_f2.8":"Canon EF 100mm f/2.8 Macro",
 "ef35mm_f2.0":"Canon EF 35mm f/2",
 "ef40mm_f2.8":"Canon EF 40mm f/2.8",
 "ef50mm_f1.8":"Canon EF 50mm f/1.8",
 "ef85mm_f1.8":"Canon EF 85mm f/1.8 USM",
 "rf16mm_f2.8":"Canon RF 16mm f/2.8 STM",
 "rf24mm_f1.8":"Canon RF 24mm f/1.8 Macro",
 "rf35mm_f1.8":"Canon RF 35mm f/1.8 Macro",
 "rf50mm_f1.8":"Canon RF 50mm f/1.8 STM",
 "sigma70mm_f2.8":"Sigma 70mm f/2.8 Macro EX DG",
 "yongnuo_50mm_f1.8":"Yongnuo 50mm f/1.8",
 "canon_ef50_f1.2l":"Canon EF 50mm f/1.2L",
 "nikon_nikkor50_f1.4_ais":"Nikon Nikkor 50mm f/1.4 AI-S",
 "zeiss_planar_t50_f1.4":"Zeiss Planar T* 50mm f/1.4",
 "leica_summicron_m50_f2":"Leica Summicron-M 50mm f/2",
 "canon_fd28_f2.8sc":"Canon FD 28mm f/2.8 S.C.",
 "nikon_nikkor28_f2.8s":"Nikon Nikkor 28mm f/2.8 S",
 "zeiss_distagon_t25_f2.8":"Zeiss Distagon T* 25mm f/2.8 Contax"}

def load(name):return json.loads((ROOT/"artifacts"/name).read_text())
def main():
 pilot=load("qfilm04_cpu_18_pilot_20261028.json")
 wave=load("qfilm04_cpu_18_wavefront_20261028.json")
 cross=load("qfilm04_cpu_11_crosscheck_20261028.json")
 airy=load("qfilm04_airy_sampling_control_20261028.json")
 assert pilot["summary"]["attempted"]==18
 assert cross["summary"]["lenses"]==11 and cross["summary"]["comparisons"]==22
 wmap={c["name"]:c for c in wave["cases"]}
 rays=[r for c in wave["cases"] for r in c["rays"] if "error" not in r]
 screens=[r for c in cross["cases"] for r in c["wavelengths"] if "max_abs_mtf_difference" in r]
 max_opd=max(r["corrected_OPD_full_film_um"]["max_abs"] for r in rays)
 max_film=max(r["film_ray_intercept_x_mm"]["max_abs"] for r in rays)
 max_efl=max(c["efl_abs_difference_mm"] for c in pilot["cases"] if c.get("import_ok"))
 rows=[]
 for i,c in enumerate(pilot["cases"],1):
  name=c["name"];n=sum("error" not in r for r in wmap[name]["rays"])
  status=("计算完成；程序判为有效" if c.get("quality",{}).get("valid") else
          "计算完成；边缘场判无效" if c.get("solver_ok") else
          "源模型瞳面初始化失败" if not c.get("import_ok") else
          "边缘缺少可用透射光线，拒绝解算")
  efl=f'{c["physical_efl_mm"]:.5f}' if c.get("import_ok") else "—"
  label=NAMES[name]
  rows.append(f'| {i} | [{label}]({c["source_url"]}) | {efl} | {n}/6 | {status} |')
 art=["qfilm04_cpu_18_import_20261028.json","qfilm04_cpu_18_pilot_20261028.json",
      "qfilm04_cpu_18_wavefront_20261028.json","qfilm04_cpu_11_crosscheck_20261028.json",
      "qfilm04_airy_sampling_control_20261028.json","qfilm04_cpu_interface_regression_20261028.json",
      "qfilm04_invalid_pupil_regression_20261028.json"]
 text=f'''# Qfilm0.4-c / Qfilm0.4-g：18 个真实产品关联公开处方的首轮验证

日期：2026-10-28。**CPU 测试已实际完成；尚未通过物理成像精度验收。**

## 版本和计算路径

`Qfilm0.4` 定义为最新的固定机械像面、真实出射平面直接标量 Huygens、Jacobian 面积振幅和复数多波长 OTF 路径。
`Qfilm0.4-c` / `-c` 使用 CPU，`Qfilm0.4-g` / `-g` 使用 CUDA GPU；物理方程相同，GPU 不可用时明确报错。
保留旧 Qfilm0.3 FFT 入口；不能把旧快速 FFT 结果改标签冒充 0.4。
当前完整 J 的照度、绿色 EE 等分量仍属暂定模型；`training_eligible=False`、`physics_certified=False`。

```bash
python -m evaluator.qfilm04 -c --prescription candidate.json --fields 3 --pupil 256 --detector 128 --detector-pitch-um 5 --detector-block 64 --output cpu.json
python -m evaluator.qfilm04 -g --prescription candidate.json --fields 3 --pupil 256 --detector 128 --detector-pitch-um 5 --output gpu.json
```

这些命令展示入口和旧网格的复现方式，**5 μm 设置不是精度推荐值**。精细网格需同时增大 detector，保持足够物理窗口。

## 实际执行范围与证据等级

| 项目 | 本轮完成结果 | 不能据此声称 |
|---|---|---|
| 公开处方导入及 550 nm EFL 核对 | 18 个候选，17 个导入；最大 EFL 差 {max_efl:.3g} mm | 处方与量产实物完全相同 |
| Qfilm0.4-c 最终评分路径 | 18 个均尝试；12 个完成中心/边缘解算，9 个程序判为有效 | 9 个通过物理精度认证 |
| 同入射光线 OPL/OPD/相位/Jacobian | 17 个可导入处方，102 组尝试，90 组具有足够共同有效样本 | 所有像场均有效、完整波场能量输运已认证 |
| 两套 Huygens 实现的 MTF 筛查 | 11 个产品关联处方 × 550/670 nm 中心场，共 22 组 | 低采样参考已收敛、差异就是实物误差 |
| Airy 解析 PSF 采样控制 | 8 组，5 μm 与 0.25 μm，固定 0.64 mm 窗口 | 实际摄影镜头等于理想 Airy PSF |
| CPU/GPU 本轮数值一致性 | CPU 入口及无 CUDA 拒绝静默回退通过；工作站 GPU 对照仍待执行 | 本轮已实测验证 -c/-g 数值一致性 |

CPU 环境：{pilot["torch_version"]}，4 计算线程（接口回归使用 2 线程）。完整 CPU pilot 使用 550 nm、36×24 mm、中心/半对角线边缘、256² 光阑网格、128²×5 μm 像面窗口。
这是单色初测，不能当作七波长摄影指标的认证。原始 `J_task` 为该试验任务的暂定评分，不适合直接跨不同焦距/光圈镜头排名。

## 18 个案例及其结果

名称表示公开文件关联的商品型号；链接指向固定版本的处方来源。多数为专利实施例或独立重建，**没有证明与工厂实际制造处方完全一致**。
[ZemaxSims 作者说明](https://github.com/Amarthgul/ZemaxSims/blob/607ac6873b34b332c36cde428c70ace22ca43370/README.md)明确指出，多数数据直接来自专利且未优化。
DeepLens 固定版本为 `911bb9d334fce7e20a4407c4506ac683b52ff246`，Zemax 来源固定版本为 `607ac6873b34b332c36cde428c70ace22ca43370`，原始字节 SHA256 保存在结果中。

| # | 产品关联公开处方 | 550 nm EFL / mm | 完成的共同光线组 | CPU 两场点结果 |
|---|---|---:|---:|---|
'''+'\n'.join(rows)+f'''

共同光线组为 3 场点（0、0.5、1）×2 波长（550、670 nm），32² 网格；共同光线不足或缺少有效 Jacobian 差分邻域时该组不成立。
“边缘无效”是当前导入/追迹/口径条件下的结果，不能直接解释为该商品实物不覆盖全画幅。
RF16 源文件本身声明该专利实施例不能覆盖全画幅。Planar 示例在上游初始化阶段得到零入瞳半径，保留为源文件/读取器异常，不擅自修改尺寸凑出结果。

## 发现一：参考波长不一致，已修复

固定版本 DeepLens 的 `DEFAULT_WAVE` 实际为 0.587 μm，原测试却将其缓存 EFL 当作 550 nm。
导入器现显式将派生 EFL、瞳面和场角归一化统一到 550 nm，并逐波长提取相同材料的折射率。
不拟合商品标称焦距，不修改源处方曲率、厚度、像面或玻璃。
修复后，17 个导入模型的 EFL 与 Qfilm 近轴计算达到上述浮点一致性。

## 发现二：共同光线几何和相位一致，不等于完整 PSF 已正确

90 个完成组的最大像面光线截距差为 {max_film:.3g} mm；去共同 piston 后的最大像面 OPD 差为 {max_opd:.3g} μm。
Jacobian 仅在双方均有效的差分邻域内比较；无效邻点不能用于宣称振幅差异。
这些结果支持相同已采样光线的折射、光程和相位实现一致；不认证统一光阑照明、入瞳到光阑的通量 Jacobian、边界离散、偏振或真实玻璃光谱。

## 发现三：5 μm 像面点采样不能作为普遍精度设置

解析圆瞳控制采用同一个 Airy 强度分布，固定 0.64 mm 窗口；直接对采样 PSF 求傅里叶值并与解析圆瞳 MTF 比较，隔离像面采样问题。
8 个控制条件（550/670 nm × f/1.4、1.8、2.8、5.6）中：

- 5 μm 点采样的最大绝对 MTF 偏差范围为 {min(c["max_abs_error_5um"] for c in airy["cases"]):.3f}–{max(c["max_abs_error_5um"] for c in airy["cases"]):.3f}。
- 0.25 μm 点采样的最大偏差范围为 {min(c["max_abs_error_0p25um"] for c in airy["cases"]):.6f}–{max(c["max_abs_error_0p25um"] for c in airy["cases"]):.6f}；有限窗口仍贡献残差。

这不是 Qfilm 对实物镜头的误差测量，也不是实物误差上界；它证明旧像面网格可以严重混叠，两个实现共享网格误差时仍可能彼此接近。
请求频率低于网格 Nyquist（例如 80<100 lp/mm）并不足够，因为被采样的光学 PSF 仍可包含更高频率。
依据圆瞳近轴截止频率 `1/(λ F/#)`，初步需要 `Δx≤λ F/#/2`；实际验收仍须独立缩小步距并扩大支持窗口。
Qfilm0.4 已输出这一**诊断估计**与 `sampling_converged=False`，且物理采样配置已纳入结果哈希，避免不同窗口/步距缓存混用。

原理参考：[Ansys FFT MTF 截止频率](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/FFT_MTF.html)；[Ansys Huygens PSF 参数与采样](https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function)。

## 发现四：22 组低光线数参考只能筛查，不能认证

该 CPU 筛查采用原生 DeepLens Huygens，每波长 16,384 条随机入瞳光线；仅将原函数射线分批从 10,000 改为 256 以控制内存，光学公式不变；中心固定在物理轴上。
最大方法间 MTF 差范围 {min(c["max_abs_mtf_difference"] for c in screens):.4f}–{max(c["max_abs_mtf_difference"] for c in screens):.4f}。
参考 4 像素边界圈能量范围 {min(c["deeplens_border_flux_fraction"] for c in screens):.3%}–{max(c["deeplens_border_flux_fraction"] for c in screens):.3%}，并且未做光线数、随机种子、传播平面和像面步距/窗口收敛。
高边界能量可能涉及随机积分噪声、有限窗口、传播波场模型或其组合，不能单独归因为窗口截断。
部分 Qfilm 红光 PSF 的边界能量同样异常，不能仅凭 `quality.valid=True` 接受。
**这 22 组全部不作为精度通过证据，也不将其方法间差异归为 Qfilm 的实物预测误差。**

## 已修复的工程问题与下一道验收门槛

1. CPU 验证模块不再在导入时强制加载 GPU 专用 Triton。
2. 统一 550 nm 派生光学参数；固定独立 Zemax 来源版本，接受 ASCII/UTF-16 原始文件。
3. 无透射瞳面、无效主光线、非有限或零 PSF 能量明确拒绝；RF16/RF24 无效瞳面回归已通过。
4. `-c` 真实 EF50 处方接口回归通过，`-g` 无 CUDA 时明确报错；没有伪造 GPU 一致性结果。
5. 物理配置哈希包括瞳面、像面步距和窗口；CLI 支持显式 `--detector-pitch-um`、`--detector-block`。

下一步顺序：先验证像面步距和支持窗口，再做入射照明/出瞳通量映射与瞳面分辨率收敛，随后提高独立参考光线数/比较确定性积分，并核查参考传播面。
目标 `MTF` 自收敛≤0.005、经独立收敛方法间差≤0.01；这只是待达到的门槛。
工作站 Bridge 最新可读取心跳仍为北京时间 2026-10-28 21:09:49，未见 Qfilm0.4 GPU 任务结果，排队不等于执行完成。
此前 18 案例七波长三场点 GPU 任务仍须在 Bridge 恢复后执行与收敛；本轮 CPU 证据不能替代这些任务。

**判断：版本命名与 CPU 可执行性已经落实，17 个产品关联公开处方已进行实际测试；当前真实摄影镜头的 PSF/MTF 精度尚未通过验收，不能进入正式训练集。**

## 原始结果

'''+''.join(f'- [{n}](../artifacts/{n})\n' for n in art)+'''
复现脚本：`validation/qfilm04_cpu_real_lens_validation.py`、`validation/qfilm04_real_lens_wavefront_audit.py`、`validation/qfilm04_cpu_camera_crosscheck.py`、`validation/qfilm04_detector_sampling_analytic.py`。
CPU 脚本通过 `--deeplens-root` / `--zmx-root` 指定固定版本源码；需将 DeepLens 放入 PYTHONPATH。报告生成器要求全部任务完成后才生成结论。
'''
 target=ROOT/"docs/QFILM04_REAL_LENS_CPU_VALIDATION_20261028.md"
 target.write_text(text)
 print(target)

if __name__=="__main__":main()
