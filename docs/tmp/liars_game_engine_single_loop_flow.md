# liars_game_engine 代码阅读：从入口到一次循环，再到结束条件

> 目标：从游戏入口开始，按代码执行顺序解释“初始化 -> 主循环一次迭代 -> 结束判定”。
> 说明：主循环我只详细拆“第 1 次迭代”，后续迭代按同样机制重复，直到触发结束条件。

## 1) 程序入口（`liars_game_engine/main.py`）

1. `main.py:52-53`：`if __name__ == "__main__": main()`
   - 你用 `python -m liars_game_engine.main` 启动时，进入 `main()`。

2. `main.py:39`：定义 `main()`。

3. `main.py:48`：`summary = asyncio.run(run())`
   - 建立事件循环并执行异步 `run()`。
   - 等 `run()` 返回一个字典摘要（对局结果）。

4. `main.py:49`：打印 `Liar's Bar run finished: {...}`。

---

## 2) 运行装配（`run()`）

5. `main.py:13`：定义异步 `run()`。

6. `main.py:22`：`settings = load_settings()`
   - 进入 `liars_game_engine/config/loader.py:32`。
   - `loader.py:42-43` 构造配置和 `.env` 路径对象。
   - `loader.py:45-47` 读取 YAML（不存在则空字典）。
   - `loader.py:49` 调用 `AppSettings.from_dict(...)` 组装强类型配置。
   - `loader.py:51` 解析 `.env`。
   - `loader.py:52-57` 用 `.env` 覆盖 API key/base_url/timeout（如果有）。
   - `loader.py:59` 返回 `settings`。

7. `main.py:23`：`env = GameEnvironment(settings)`
   - 进入 `liars_game_engine/engine/environment.py:18`。
   - `environment.py:27-31` 保存配置、创建随机数器、初始化声明/挑战/轮盘规则模块。
   - `environment.py:33-36` 取玩家 ID，校验人数必须 2~4，随机打乱座位顺序。
   - `environment.py:38-45` 为每名玩家创建 `PlayerRuntimeState`，并生成各自左轮牌堆。
   - `environment.py:47` 随机选首轮起手玩家。
   - `environment.py:48-56` 构建 `RuntimeGameState`：
     - 设置 `turn_order/current_player_id/round_index/phase/table_type` 等。
   - `environment.py:57` `_deal_round_cards()` 发首轮手牌：
     - `environment.py:97-108` 给每位存活玩家发 `cards_per_player`（默认 5）张。

8. `main.py:24`：`agents = build_agents(settings)`
   - 进入 `liars_game_engine/agents/factory.py:9`。
   - `factory.py:20-40` 遍历配置玩家：
     - `agent_type == "langchain"` 则建 `LangChainAgent`。
     - 否则建 `MockAgent`（本项目默认配置是 `mock`）。
   - 返回 `{player_id: agent}` 映射。

9. `main.py:26`：生成 `game_id`（如 `game-20260319-121502`）。

10. `main.py:27`：创建 `ExperimentLogger`。
    - `experiment/logger.py:19-23` 创建日志目录并确定 JSONL 文件路径。

11. `main.py:29-35`：构建 `GameOrchestrator`，注入：
    - `env`、`agents`、`logger`、`fallback_action`、`max_turns`。

12. `main.py:36`：`return await orchestrator.run_game_loop()` 进入主循环。

---

## 3) 主循环：第 1 次迭代逐行（核心）

> 文件：`liars_game_engine/experiment/orchestrator.py`。

13. `orchestrator.py:47`：`turns_played = 0`。

14. `orchestrator.py:49`：进入 `while turns_played < max_turns and not env.is_game_over()`。
    - 左条件：回合数没到上限。
    - 右条件：游戏未结束。
    - `env.is_game_over()` 在 `environment.py:229-240`：
      - 若 `phase == GAME_OVER` 则结束。
      - 否则只要存活人数 `<=1`（`_alive_count()`）也结束。

15. `orchestrator.py:50`：`player_id = env.get_current_player()`。
    - 对应 `environment.py:125-134`，返回当前行动玩家。

16. `orchestrator.py:51-52`：如果当前玩家没有对应 agent，直接 `break`（保护逻辑）。

17. `orchestrator.py:54`：取出该玩家 agent 实例。

18. `orchestrator.py:55`：`observation = env.get_observation_for(player_id)`。
    - `environment.py:145-171` 组装观察：
      - 当前 phase、当前玩家、存活玩家。
      - 当前玩家私有手牌 `private_hand`。
      - 每个玩家手牌数量（公开计数）`public_card_counts`。
      - 待挑战声明 `pending_claim`（若存在）。
      - 当前轮次/回合索引/table_type/是否本轮第一手。
      - `must_call_liar`（是否被规则强制必须 challenge）。

19. `orchestrator.py:56`：`decision = await agent.act(observation)`。
    - 默认是 `MockAgent.act()`（`agents/mock_agent.py:27-75`）。
    - 第 1 轮通常 `phase=turn_start` 且有手牌，所以走“出牌”分支：
      - `mock_agent.py:51-55` 选 1~3 张要打出的牌、决定宣称点数。
      - `mock_agent.py:56-69` 返回 `ActionModel(type="play_claim", ...)`。

20. `orchestrator.py:58`：`step_result = env.step(player_id, decision.action)`。
    - 进入统一动作入口 `environment.py:445-474`：
      - `455-457` 若游戏已结束，拒绝动作。
      - `459-460` 非当前玩家动作，拒绝。
      - `462-463` 已淘汰玩家动作，拒绝。
      - `465-466` 若动作是 `play_claim`，进入 `_handle_play_claim()`。

21. `_handle_play_claim` 执行（`environment.py:326-373`）：
    - `336`：先用 `DeclareRule.validate(...)` 校验。
      - `declare_rule.py:23-59` 校验 phase、出牌数(1~3)、玩家手牌是否真的拥有这些牌。
    - `340-341`：若规则要求“只剩你有牌，必须 call LIAR”，则拒绝继续出牌。
    - `343-348`：从玩家真实手牌中移除本次打出的牌。
    - `349-354`：把本次声明写入 `pending_claim`。
    - `355`：phase 切到 `RESPONSE_WINDOW`（给下家响应/挑战窗口）。
    - `356-357`：标记不再是本轮第一手，回合计数 `turn_index += 1`。
    - `359`：找下一个“未淘汰且有手牌”的玩家。
    - `360-363`：记录本次事件文本（出牌数量、针对桌牌类型）。
    - `365-368`：若没人可响应，直接开新一轮。
    - `370-372`：否则把 `current_player_id` 交给下一个响应者并返回成功。

22. `orchestrator.py:59-60`：先初始化回退状态：
    - `fallback_used = False`
    - `fallback_reason = None`

23. `orchestrator.py:62-65`：如果 `step_result.success == False`：
    - 标记用了回退动作。
    - 用 `ActionModel(type=fallback_action)` 再执行一次。
    - 当前默认 `fallback_action` 是 `challenge`（来自配置）。
    - 正常首轮一般不会触发这段。

24. `orchestrator.py:67`：生成本回合追踪 ID（`turn-1-xxxxxxxx`）。

25. `orchestrator.py:68-98`：`logger.record_turn(...)` 写日志。
    - 记录 observation、thought、action、raw_output、parser_error、fallback 信息、step_result。
    - `experiment/logger.py:33-36` 自动补 UTC 时间戳，按 JSONL 追加写入。

26. `orchestrator.py:100`：`turns_played += 1`。
    - 第 1 次循环结束。

---

## 4) 后续循环如何推进到游戏结束

27. 下一次回到 `orchestrator.py:49` 继续判断 while。

28. 后续每轮仍是同模板：
    - 取当前玩家 -> 取 observation -> agent 产出动作 -> `env.step` 结算 -> 记日志 -> `turns_played+1`。

29. 当某次动作是 `challenge` 时，会走 `environment.py:374-420`：
    - 校验 challenge 合法性。
    - 判定被质疑牌里是否有 Liar（`challenge_rule.py:40-55`，非桌牌且非 `JOKER` 即 Liar）。
    - 有 Liar：被挑战者受罚；无 Liar：挑战者受罚。
    - 受罚方式：俄罗斯轮盘（`roulette_rule.py:38-65`），翻到 `LETHAL` 就淘汰。
    - 若存活人数 `<=1`，直接把 phase 设为 `GAME_OVER`。
    - 否则按规则开新一轮（新 table_type、重发牌、确定起手）。

30. while 退出条件有两个（任一满足即退出）：
    - `turns_played >= max_turns`，或
    - `env.is_game_over() == True`。

---

## 5) 循环退出后的收尾（胜者与摘要）

31. `orchestrator.py:102-106`：统计所有未淘汰玩家 `alive_players`。

32. `orchestrator.py:107`：若恰好 1 人存活，则该玩家为 `winner`；否则 `winner=None`。

33. `orchestrator.py:108-113`：返回摘要字典：
    - `turns_played`
    - `game_over`
    - `winner`
    - `log_file`

34. 回到 `main.py:48-49`：打印最终摘要并结束进程。

---

## 6) 一条真实运行样例（便于对照）

我刚跑了一局本地 mock 配置，结果：

- 最终摘要：`turns_played=26, game_over=True, winner='p1'`
- 日志文件：`logs/runs/game-20260319-121502.jsonl`
- 该局第 1 回合里：
  - 当前玩家 `p2`
  - `p2` 打出 1 张牌并声称 `A`
  - 环境返回成功，响应窗口交给 `p1`

这和上面“第 1 次迭代逐行”完全一致。
