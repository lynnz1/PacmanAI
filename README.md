# Pacman AI Project (更新于 2025年5月)

该项目为大学期间参与的 AI 吃豆人智能体训练竞赛作品，旨在探索不同算法在博弈环境中的表现。使用 Python 实现了 Q-Learning、A*、蒙特卡洛搜索等算法，模拟智能体在游戏中进行路径规划与策略决策。

## 项目成果
- 第一版本基于 Q-Learning 和 A* 控制两个吃豆人智能体，在 180 局对战中胜率为 **45%**
- 通过引入行为决策树并优化路径规划策略后，最终版本胜率提升至 **62%**
- 在约 70 支参赛队伍中，最终排名第 21

## 使用算法简介
- **Q-Learning**：基于贝尔曼方程计算 Q 值，通过训练智能体更新行为策略
- **A\***：结合启发式函数为吃豆人规划最短路径，适配多目标选择
- **决策树**：在运行时根据当前游戏状态动态选择最优行为模式

---

📂 项目为归档展示用，代码结构清晰，欢迎参考学习。

# PacmanAI
![preliminary_contest](https://user-images.githubusercontent.com/128367217/226707426-2fb8688b-7735-445b-bd92-961945731cfe.gif)
##### （A* 算法在offensiveAstar（）里面)

##### A* 的goal state如下：

食物goal：
class SearchFoodCapusule(PositionSearchProblem)

加速球：
class SearchCapsule(PositionSearchProblem)

场地中线：
class SearchEscape(PositionSearchProblem):

家：
class SearchHome(PositionSearchProblem): 


##### Q-Learning训练前的表现如下：
![preliminary_contest](https://user-images.githubusercontent.com/128367217/226707426-2fb8688b-7735-445b-bd92-961945731cfe.gif)
##### Q-Learning训练后的表现如下：

![68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d31416b713175464439654e41574b756253716e316a70485f77442d5f6b74314146](https://user-images.githubusercontent.com/128367217/226709276-e9c33b41-3401-44df-beab-3afd584168ed.gif)
