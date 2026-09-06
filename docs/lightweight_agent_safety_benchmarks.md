# 轻量 Agent 隐私与安全 Benchmark 调研

日期：2026-09-06。筛选标准：可用本地小模型运行、具备公开代码/数据、能映射到
`task_completed`、`privacy_leak`、`attack_success` 三类指标中的至少两类，并且不要求
真实外部账户或不可控在线服务。

## InjecAgent 的关系

InjecAgent 是一个独立 benchmark，并不在其仓库中调用 AgentDojo、ToolEmu 或其他
benchmark。其 README 明确提供自身的 1,054 个 DH/DS 工具注入 case、prompted-agent
与 fine-tuned-agent 两条评测入口。它适合作为**工具注入与两阶段隐私泄露**的受控环境，
但不应被当作跨环境泛化结论。

## 候选项

| Benchmark | 安全/隐私覆盖 | 环境与轻量性 | 与当前架构的接入方式 | 推荐级别 |
|---|---|---|---|---|
| AgentDojo | 间接提示注入、越权工具调用；本项目已有 forbidden-source 隐私泄露映射 | 本地、状态化工具环境；现已接通 Agent-R1 defender | 直接复用 `AgentEnv` adapter 与 frozen attacker 数据 | 已接入，主基准 |
| InjecAgent | 间接注入；DH 与读取→外发的 DS 隐私链 | 本地模拟工具、1–2 step；最轻量 | 复用现有 renderer/metrics；适合 attacker curriculum 与快速回归 | 已接入，主基准 |
| BIPIA | 间接提示注入与防御鲁棒性 | 主要是文本/应用任务，不执行真实工具；CPU 即可构造样本 | 新增 text-observation adapter；将泄露/服从目标作为规则判分 | 首选新增：低成本泛化集 |
| ToolEmu | 工具代理风险（含隐私、金融、通信等高风险动作） | LLM 模拟工具与 evaluator；不需要真实服务，但每条 rollout 比前两者慢 | 项目已有 PrivacyLens/ToolEmu 相关代码；统一为允许操作、敏感操作、任务完成三元组 | 次选：更真实的工具风险 |
| Agent Security Bench (ASB) | 多类 agent 攻击与安全失效 | 官方实现规模较大，部分场景依赖更复杂的环境/代理配置 | 先只抽取纯 tool-use / prompt-injection 子集；不建议直接纳入 PPO 主循环 | 调研/评测用，不是第一接入项 |
| SafeAgentBench | 具身 agent 的安全任务规划 | 依赖具身仿真任务，资源与适配成本较高 | 只能迁移其安全规划指标；不适合作为当前文本工具 agent 的 rollout 环境 | 不推荐当前接入 |

## 推荐实验顺序

1. **AgentDojo + InjecAgent**：保留为训练环境，分别覆盖多步状态工具与两阶段泄露。
2. **BIPIA**：先实现小规模、无工具的 held-out 泛化评测；它不进入 PPO reward，避免
   为单一文本格式过拟合。
3. **ToolEmu / PrivacyLens**：作为第二个训练或评测环境，使用模拟工具来检验“最小权限、
   敏感操作拒绝、任务完成”能否迁移。
4. **ASB**：只在计算预算充足后选择与当前 API/tool calling 接近的子任务做外部报告。

## 统一接口建议

每个 adapter 只需导出：

```python
{
    "task_completed": float,       # 合法用户任务完成
    "privacy_leak": float,         # 敏感数据访问/外发或等价违规
    "attack_success": float,       # 注入/越权目标被执行
    "allowed_actions": list[str],  # 可选，供通用 defender 模板使用
    "sensitive_actions": list[str] # 可选，供隐私边界判分使用
}
```

这样 attacker 的结构化内容 reward 与 defender 的 trust-boundary 模板不依赖任何一个
benchmark 的攻击工具名称；只有环境结果项由各 adapter 填充。

## 公开来源

- InjecAgent: https://github.com/uiuc-kang-lab/InjecAgent
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- BIPIA: https://github.com/microsoft/BIPIA
- ToolEmu: https://github.com/ryoungj/ToolEmu
- Agent Security Bench: https://github.com/agiresearch/ASB
- SafeAgentBench: https://github.com/shengyin1224/SafeAgentBench
