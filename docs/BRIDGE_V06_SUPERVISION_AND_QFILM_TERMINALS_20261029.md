# Bridge v0.6 与 Qfilm 三个工况的终态验收

## 本轮改动

代码提交 `d22f69ba6ec69ae0921a24c125b701ad33714fc0`。Bridge 的网络读取和发布由后台线程处理；计算监管循环只检查本地控制快照、真实子进程和本地退出记录。GitHub 网络错误不再使正在运行的子进程失去监管。

- 持久化上传队列按目标路径合并，按数据观测时间保留新值；远端已有更新观测时不覆盖；终态记录不回退为 running。
- 网络同步失败时，已运行的任务继续受本地超时和退出码监管；控制快照过期时暂停启动新任务。
- 新 worker 包装器保存实际子进程身份、独立进程组、超时/停止原因和真实退出码。
- 启动时观察既有计算进程、持续保存/发布其检查点；对无法获取历史退出码的旧进程，标为 recovered_unknown_exit，不伪造 finished/0。
- 单实例文件锁、原有持久化 claim 与新任务 ID 策略保留；现有目录不清除、不重复执行已领取的 ID。
- 旧版正在运行的三个工况任务优先被观察，避免与新完整复算同时抢占计算资源。其他旧任务的原始产物保留。

这些措施容忍网络中断，并不等于已经修复工作站外部网络/TLS 路径。代理、路由等仍需按实际网络证据定位；未关闭 TLS 校验。

## 验证

`scripts/test_bridge_v06.py` 的 **13 项测试通过**，包括真实子进程在模拟断网期间继续运行、保存非零退出码、上传失败保留数据、旧心跳不覆盖新观测、终态防回退、已领取任务不重复执行、预置请求仍经过完整原有验证，以及独立终态校验器对缺失/伪造/放宽门槛的拒绝。

另有本地部署模拟：完整复制运行时、执行测试、备份检查点并保留源文件，在无管理权限时返回 staged_requires_host_activation；该模拟不证明工作站已有安装权限。记录：`artifacts/bridge_v06_validation_20261029.json` 与 `artifacts/bridge_v06_stage_local_test_20261029.json`。

## 安装与进程保护

优先任务：`000-000-000-000-000-bridge-v06-stage-20261029-01`。

安装任务会在工作站实际运行测试，备份项目 JSON 检查点，生成 SHA256 清单，把稳定运行时放在：

`/var/lib/cots-lamcts/runtime/bridge-v06-d22f69ba6ec6/`

如果当前服务允许非交互式 sudo，将调度受控激活。若服务仍受只读系统目录或 NoNewPrivileges 限制，则返回 `staged_requires_host_activation`，需要管理员在本机执行已经准备好的命令：

```bash
sudo /usr/bin/python3 /var/lib/cots-lamcts/runtime/bridge-v06-d22f69ba6ec6/scripts/deploy_bridge_v06.py --activate
```

**该命令只适用于工作站已完成 staging 的情况；不能把代码已提交视为文件已在工作站落地。** 安装报告位于该任务产物 `bridge-v06-deployment.json`，其中给出实际路径和激活命令。

激活脚本检查全部运行时文件的 SHA256，再次运行测试，写 systemd drop-in，确认最终 ExecStart 和 `KillMode=process`，然后仅替换服务主进程。既有计算子进程保留，由新版本观察；不会使用 pkill、清除回执或删除原始检查点。基础服务的权限限制仍保留。

新版本的实际启动证明为 `remote/health/bridge-v06-start.json`；仅看到旧版本 agent_status 不能视为升级完成。

## 三个完整工况

如果旧 Bridge 的网络重试或上传积压导致 staging 任务无法执行，另提供独立的本机管理员恢复入口 `scripts/bootstrap_bridge_v06_host.sh`。先下载并审阅该脚本，在工作站运行：

```bash
sudo bash bootstrap_bridge_v06_host.sh
```

该脚本仅下载固定提交的 8 个运行时文件，逐个核对预置 SHA256，再执行上述测试、检查点备份与受控激活；不读取 GitHub token，不放宽 TLS 验证，不删除旧任务。下载失败会停止，不会安装未校验的代码。已有 staging 时可直接使用安装报告中的激活命令。

| 工况 | 新的不可重复任务 ID |
|---|---|
| EF40/430 nm、中心场 | 000-000-011-qfilm06-ef40-blue-20261029-01 |
| EF40/550 nm、中心场 | 000-000-012-qfilm06-ef40-green-20261029-01 |
| RF50/670 nm、中心场 | 000-000-013-qfilm06-rf50-red-20261029-01 |

请求放在 `remote/prepared/bridge-v06/`，由 v0.6 在激活后读取。旧 v0.5 不会提前消费这些请求。执行入口再次验证 v0.6 的进程身份，缺少可靠监管时不启动计算。

Qfilm 的波动传播方程、0.25 μm 默认采样、冻结 CPU/GPU 实际网格序列及现有收敛门槛不变。若旧尝试已产生完整且独立验收通过的结果，明确记录复用来源与执行 ID，不冒充一次新的 GPU 执行；否则采用新 ID 复算。

独立校验器 `infra/qfilm_convergence_status.py` 会读取每个工况的 CPU/GPU 原生终态，验证处方来源、波长/场点、采样计划哈希、实际设备和执行 ID，并从保存的复 OTF 与原始积分重新计算：

- 瞳面加密、扩窗、步距减半、联合提升的差值；
- CPU/GPU 所有对应网格的复 OTF 与积分差值；
- 相位初筛和原定验收门槛没有被放宽。

没有终态、资源耗尽或任一收敛检查失败，都不能计为通过。旧程序退出码 0、Bridge finished、小窗口算术一致也不能替代此验收。

汇总发布到 `remote/health/qfilm-three-verification.json`。只有 `all_three_verified=true` 才能声称三个工况均已完整通过；`training_eligible=false` 仍保留，因为独立物理模型参考及实物标定是另外的门槛。

## 当前发布与实际部署须区分

本文件描述已实现、已测试、已提交的升级与验收流程。工作站是否已安装、是否已切换 v0.6，以及三个工况是否已取得完整终态，必须以新任务回传、启动证明和独立验收报告为准。
