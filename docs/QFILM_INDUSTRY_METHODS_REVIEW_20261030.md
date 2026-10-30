# Qfilm 的行业计算方案与技术路线调研

检索日期：2026-10-30。范围：成像镜头处方 → 多视场、多波长 PSF/OTF/MTF → 成像评价与搜索评分。以厂商技术文档、项目官方文档、作者论文为依据；“行业通用”指这些成熟工具公开提供的方法，并非市场份额排名。下述接入优先级属于针对 Qfilm 的工程判断，尚不是新求解器实测结论。

## 1. 判断

Qfilm 当前的大规模逐源点、逐像点直接积分可保留为参考内核，但不宜作为每个搜索候选的默认计算路径。成熟成像工具组合使用光线追迹、复波前、傅里叶衍射、指定频率 MTF、Huygens 或更完整的物理光学分析。FFT 是数值计算工具；是否可靠取决于物理模型、坐标、振幅/OPD、采样和边界条件，而不是方法名字。[S01,S02,S03]

0.25 µm 是局部 PSF 的像面输出步长。它不是要求整个 36×24 mm 像面都按该步长建立一张网格，也不要求输出每个像素时都遍历全部出瞳源点。固定步长的输出可由 MFT、CZT 或适当的缩放传播实现，但不能因此忽略出瞳采样和混叠条件。[S08,S09,S10]

## 2. 成熟软件提供什么

| 工具 | 已确认的方法/能力 | 对 Qfilm 的角色判断 |
|---|---|---|
| Zemax OpticStudio | FFT PSF/MTF、Huygens PSF/MTF；ZOS-API 可提取分析数组；官方有 Python MTF 示例 | 优先作为处方和多视场结果的独立商业参考，也可通过 API 承担评价后端。[S01,S03,S04] |
| CODE V | 光线追迹、PSF、衍射 MTF；Gaussian Beam Trace、FFT beam propagation、Beam Synthesis Propagation；COM 接口 | 另一成像设计参考；复杂波传播可评估 BSP，不能把所有 CODE V 成像分析都等同 BSP。[S05,S06] |
| OSLO | FFT 或直接积分 PSF、Kirchhoff/RS 积分、矢量衍射、FFT/卷积 MTF | 明确支持按问题选择数值方法，可作为另一套成像参考。[S07] |
| VirtualLab Fusion | Ray Tracing 与 Field Tracing；不同尺度模型组合；支持成像 PSF/MTF；2026.1 增加 CODE V SEQ 导入 | 更适合复杂物理光学或混合折射/衍射系统，不必作为当前常规 COTS 镜头搜索的第一接入目标。[S15] |
| FRED | 非序列系统分析；Gaussian Beam Decomposition 传播相干场 | 复杂光路、干涉、杂散光与相干传播的候选；光阑效应还需注意场重构/重采样。[S16,S17] |
| COMSOL Wave/Ray Optics | 光线、波动、全波和包络方法及耦合 | 适合局部严格解和多物理验证；常规全波 Maxwell 方法对大型镜头的波长级网格代价很高。[S18] |

商业参考节点的部署、API 功能、并发和许可证应按实际版本确认。不能把一套成熟软件的输出直接视为实物真值；同样必须验证采样和模型设置。

## 3. 可替换当前直接求和的几条数值路线

### 3.1 光线追迹 + 复出瞳 + Fourier PSF/MTF

在合适的参考面/参考球面构造复瞳函数：

P_lambda(u,v) = A_lambda(u,v) exp(i 2 pi W_lambda(u,v)/lambda).

这里 W 是正确参考下的 OPD，A 含有效通光、振幅和合适的面积/角度权重。满足相应衍射近似时，通过傅里叶关系求焦区场与 PSF。OpticStudio 的 FFT 模型对出瞳像差、像面方向、标量近似等有适用限制；不能因为镜头名义 f/# 足够大就推断所有边缘视场都适用。[S01]

可选数值实现：

- FFT：规则网格、自然输出采样下直接高效。
- Matrix Fourier Transform / Matrix DFT：用两次可分离矩阵乘法，指定输出采样和范围。
- CZT / Bluestein：在变换模型成立时，灵活计算所需的采样区间。

POPPY 用 MFT 输出固定 Detector 采样；prysm 的当前传播文档提供 matrix-DFT/CZT 任意采样和 angular-spectrum 方法。[S08,S09]

当输入输出均为 N×N 且可使用可分离傅里叶关系时，朴素二维源点×像点求和约 O(N^4)，两步矩阵 DFT 约 O(N^3)，FFT 约 O(N^2 log N)。这些复杂度不能直接作为实际加速倍数，也不适用于任意球面核的直接替换。

### 3.2 只计算评分需要的 MTF 频率

OpticStudio 官方将指定空间频率的稀疏采样 MTF 用于优化，并与完整曲线的网格计算区分。这一点非常适合目前只取若干频率的 Qfilm 评分。[S02]

在局部非相干 Fourier 成像模型适用时，OTF 可以由复瞳的归一化自相关得到；不必为了十个频率点先生成数百万像素的 PSF。多波长组合必须保留物理位置和复 OTF 相位后再取模，不能直接平均单色 MTF。

这一分支只能替代与其数学定义相同的评分计算。完整 PSF、包围能量、窗口截断检查以及用户要求的 0.25 µm 图像验证仍保留在验证模式。

### 3.3 平面场传播：ASM、带限 ASM、缩放/偏移传播与 FFT 型 RS

对已知参考平面上的复场，在均匀介质中使用角谱传播；在数值实现上检查带限、窗口、零填充和混叠。普通同网格 ASM 不自动满足任意输入/输出步长或离轴窗口，必要时使用经验证的缩放/偏移变体。Chromatix 官方已有 scaled/shifted BLAS 与 CZT 示例。[S10]

Matsushima 与 Shimobaba 的带限角谱论文是这一路线的经典参考；它针对角谱数值传播的采样问题，不是对整套镜头几何光学模型的认证。[S12]

DeepLens 当前 HybridLens 示例采用相干光线追迹构造场，再由 ASM 传播到传感器；其对应 2024 ray-wave 论文和代码可作为接入研究对象。新主线代码与我们已锁定的旧提交不同，不能未经审计直接假定新版本修复了当前 11 处方的问题。[S13,S14]

### 3.4 NUFFT：处理非均匀采样的计算组件

FINUFFT/cuFINUFFT 支持指定数值精度的非均匀傅里叶变换及 CPU/GPU 实现。[S11]

它适合可重写为傅里叶和的非均匀出瞳/方向余弦采样问题。但当前 exp(i k r)/r 及斜入射权重不是天然的平面傅里叶核；必须先推导有效的传播表述、积分权重和误差范围。NUFFT 的数值容差不等于整个光学模型的物理误差，不能把库函数替换等同于完成光学验证。

### 3.5 Beamlet / BSP / GBD

这类方法把场表示为局部束元，独立传播后相干合成。CODE V BSP 和 FRED GBD 是公开可确认的商业实现。[S06,S16]

优点是可结合复杂光路传播。边界截断和光阑需要特别处理；FRED 官方例子明确说明了仅裁剪携带场的几何光线可能漏掉光阑对场的作用，并通过场重构处理。[S17]

### 3.6 Huygens、矢量衍射和全波

Huygens/直接积分适合不满足简化 Fourier 条件的工况及参考核验；它仍需要正确采样和模型。OpticStudio 的 Huygens 实现也不应与当前 Qfilm 核仅按名字视为逐项等价。[S01,S03]

高 NA 或偏振敏感任务可考虑矢量场模型；微纳结构和局部散射可用全波方法。常规完整 Maxwell 网格随光学尺寸和波长尺度增长很快，所以它不是当前厘米级多片镜头的大规模搜索首选。[S18]

## 4. 开源组件的接入优先级（工程判断）

| 组件 | 可利用部分 | 集成边界 |
|---|---|---|
| DeepLens / 当前 Qfilm 几何部分 | 材料、折射、色散、几何量、相干波前 | 继续审计 OPL/OPD、出瞳映射、振幅及失效模式；不把开源输出当实物真值 |
| prysm | MFT/CZT、Fourier PSF/OTF、角谱；可切换 CuPy 后端 | 我建议优先做独立传播交叉对照；它不能代替对整套镜头输入及复杂波前映射的验证。[S09,S19] |
| POPPY | 固定 detector 采样的 MFT；成熟的望远镜物理光学建模参考 | 官方明确不把它定位为 Zemax/CODE V 式完整镜头设计软件替代品。[S08] |
| Chromatix / Diffractio | 缩放/离轴/带限传播；RS、CZT 等 | 作为传播方法参考和独立算例平台；仍需接入真实镜头的复场。[S10,S20] |
| FINUFFT / cuFINUFFT | 非规则采样下的快速 Fourier 求和 | 在数学核和权重明确后考虑，属于数值组件而非完整镜头求解器。[S11] |

不能仅根据厂商或开源项目某个算例的宣传速度，给 Qfilm 直接承诺秒级或固定加速倍数。现有 11 处方、三个视场、七波长应作为接入后的实测矩阵。

## 5. 针对 COTS / LA-MCTS 的建议

建议将 Qfilm 保留为统一的评价器接口，内部采用三层执行路径：

1. **可行性与粗筛**：几何约束、遮挡、通光、焦距/光圈、像面和大像差指标；几何 MTF 可用于明确标记的粗排序，不作为最终衍射 J。
2. **日常衍射评分**：逐视场/波长构造可信复瞳；适用时使用目标频率 OTF/MTF；需要 PSF 时采用 FFT/MFT/CZT 或合适的平面场传播。主光线大角度、瞳畸变、截光及不规则映射不能默默沿用简单模型。
3. **独立验收**：针对候选优解、异常工况和跨模型边界，运行完整 0.25 µm PSF、pupil/window/pitch 收敛及独立参考比较。Huygens/直接积分保留为按范围声明的参考，不把 CPU 全量复算放在每个 GPU 任务的必经路径。

对大参数变化的 COTS 搜索，每个候选仍需检查快速模型的适用条件；不能用某一个近轴镜头通过验证就认证全部搜索空间。LA-MCTS 本身不要求光学求解器可微，因此 GPU 后端不必只用 PyTorch eager；可评估 CuPy/FFT 库或编译核，同时保留未来梯度优化所需的独立接口。

近期最合适的接入验证组合（建议）：**现有光线/材料模型 + prysm 的固定采样 Fourier 传播对照 + 经验证的缩放/偏移 ASM 分支 + Zemax/CODE V 独立处方对照**。稀疏频率评分优先；不能通过适用条件和收敛检查的结果继续返回明确失效，不能混入正式标签。

## 6. 接入时必须对齐的定义

同一处方比较时，对齐玻璃 n(lambda)、非球面项定义、长度单位、物方位置/角度、光阑、参考面、固定像面、焦点策略、实际像面坐标、通光权重、波长权重和 MTF 方向。多色图像在共同物理像面上进行非相干强度叠加，并保留色差位移。标量场的归一化不等于完成辐射通量标定；软件间一致也不等于实物校准。

最低测试集应包括：Airy、已知复瞳/离焦/像差、能量与位移关系、非对称/截光 pupil、离轴固定像面、多色位移，再进入 11 处方矩阵。CPU/GPU 应检查相同算子、网格和约定下的一致性，不能把极小的设备间误差解释为相同数量级的实物预测精度。

## 7. 主要资料（均于 2026-10-30 核查）

- [S01] [Ansys：FFT 与 Huygens PSF 的差别](https://optics.ansys.com/hc/en-us/articles/42661795535251-What-is-the-difference-between-the-FFT-and-Huygens-PSF)
- [S02] [Ansys：MTF 分析方法，含指定频率稀疏采样与几何 MTF](https://optics.ansys.com/hc/en-us/articles/42661772706835-Methods-for-analyzing-MTF-in-OpticStudio)
- [S03] [OpticStudio Help：Huygens MTF](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_MTF.html)
- [S04] [Ansys：ZOS-API 概览](https://optics.ansys.com/hc/en-us/articles/42661790380179-ZOS-API-NET-An-Overview)；[Python/其他语言示例](https://optics.ansys.com/hc/en-us/articles/42661773562899-Sample-code-for-ZOS-API-users)
- [S05] [Keysight：CODE V 功能及接口](https://www.keysight.com/gb/en/products/software/optical-solutions-software/optical-design-solutions/codev.html)
- [S06] [Keysight：CODE V BSP 物理光学方法](https://www.keysight.com/blogs/en/tech/sim-des/accurate-physical-optics-modeling-code-v-beam-synthesis-propagation)
- [S07] [Lambda Research：OSLO Standard Analysis](https://lambdares.com/support-posts/oslo-standard-analysis)
- [S08] [POPPY：算法、固定 Detector 采样与适用范围](https://poppy-optics.readthedocs.io/en/latest/overview.html)
- [S09] [prysm：传播模块，含 matrix-DFT/CZT 和 angular spectrum](https://prysm.readthedocs.io/en/latest/api/propagation.html)
- [S10] [Chromatix：Scaled and Shifted Free-Space Propagation](https://chromatix.readthedocs.io/en/latest/examples/rescaled_propagation/)
- [S11] [FINUFFT 官方文档](https://finufft.readthedocs.io/en/latest/)
- [S12] Matsushima & Shimobaba, 2009, *Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields*, Optics Express 17, 19662–19673. DOI: [10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662). [作者机构记录](https://kansai-u.repo.nii.ac.jp/records/11127)
- [S13] Yang et al., 2024, *End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model*, SIGGRAPH Asia. [作者项目页及论文](https://vccimaging.org/Publications/Yang2024HybridLens/)
- [S14] [DeepLens 官方 HybridLens ray-wave/ASM 示例](https://github.com/vccimaging/DeepLens/blob/main/0_hello_hybridlens.py)
- [S15] [LightTrans：VirtualLab Fusion 2026.1 官方功能及模型组合说明](https://www.lighttrans.com/docs/release-notes/v2026.1/)
- [S16] [Photon Engineering：FRED Gaussian Beam Decomposition 示例](https://support.photonengr.com/article/109-poissons-spot)
- [S17] [Photon Engineering：光阑和场重构的限制及处理](https://support.photonengr.com/article/228-interferometer-with-field-synthesis)
- [S18] [COMSOL：镜头的光线、BPM 和全波方法比较](https://www.comsol.com/blogs/how-to-perform-lens-simulations-using-the-wave-optics-module/)
- [S19] [prysm：GPU 与可切换数值后端](https://prysm.readthedocs.io/en/latest/how-tos/GPU%20and%20Exascale%20Computing.html)
- [S20] [Diffractio：FFT、RS、CZT 等算法](https://diffractio.readthedocs.io/en/latest/source/tutorial_scalar/algorithms/tutorial.html)
- [S21] Soummer et al., 2007, *Fast computation of Lyot-style coronagraph propagation*, Optics Express 15, 15935–15951. DOI: [10.1364/OE.15.015935](https://doi.org/10.1364/OE.15.015935). [作者预印本](https://arxiv.org/abs/0711.0368). 此论文的速度与存储结果针对其日冕仪问题，不能直接移植为 Qfilm 的提速承诺。

本轮是技术路线检索和方案评估；传播内核替换后的准确性、适用范围与实际吞吐仍需通过上述对照测试建立。
