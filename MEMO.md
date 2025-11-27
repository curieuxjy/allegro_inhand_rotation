## Dual Hand CAN Bus Configuration

**중요**: 양손 환경에서 CAN bus 매핑이 실제 하드웨어 연결과 일치해야 합니다.

`allegro_hand_bringup/config/v4/dual_hand/dual_hand.urdf.xacro` 파일에서:
- 실제 `can0`에 연결된 손 → `io_interface_descriptor="can:can0"`
- 실제 `can1`에 연결된 손 → `io_interface_descriptor="can:can1"`

현재 설정 (실제 하드웨어 기준):
- `can0` → 왼손 (Left)
- `can1` → 오른손 (Right)

```xml
<xacro:allegro_hand_ros2_control name="AllegroHandV4Right" ... io_interface_descriptor="can:can1" .../>
<xacro:allegro_hand_ros2_control name="AllegroHandV4Left" ... io_interface_descriptor="can:can0" .../>
```

**주의**: 매핑이 잘못되면 오른손 명령이 왼손으로, 왼손 명령이 오른손으로 전송됩니다!

---

 task/ Directory Dependencies

  RightCorlAllegroHandHora.yaml  ◀━━━ [BASE - Root of all configurations]
  │
  ├──▶ RightAllegroHandHora.yaml
  │    └──▶ RightAllegroHandGrasp.yaml
  │
  ├──▶ RightTipAllegroHandHora.yaml
  │    └──▶ RightTipAllegroHandGrasp.yaml
  │
  ├──▶ RightCorlAllegroHandGrasp.yaml
  │
  ├──▶ LeftCorlAllegroHandHora.yaml
  │    └──▶ LeftCorlAllegroHandGrasp.yaml
  │
  ├──▶ LeftAllegroHandHora.yaml
  │    └──▶ LeftAllegroHandGrasp.yaml
  │
  └──▶ LeftTipAllegroHandHora.yaml
       └──▶ LeftTipAllegroHandGrasp.yaml

  ---
  train/ Directory Dependencies

  RightCorlAllegroHandHora.yaml  ◀━━━ [BASE - Root of all training configurations]
  │
  ├──▶ RightAllegroHandHora.yaml
  ├──▶ RightTipAllegroHandHora.yaml
  ├──▶ RightCorlAllegroHandGrasp.yaml
  ├──▶ RightAllegroHandGrasp.yaml
  ├──▶ RightTipAllegroHandGrasp.yaml
  ├──▶ LeftCorlAllegroHandHora.yaml
  ├──▶ LeftCorlAllegroHandGrasp.yaml
  ├──▶ LeftAllegroHandHora.yaml
  ├──▶ LeftAllegroHandGrasp.yaml
  └──▶ LeftTipAllegroHandGrasp.yaml

---

## ActorCritic Network Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ActorCritic                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 1. Privileged Info Encoder (env_mlp) - Stage 1 & 2              │    │
│  │    Input: priv_info (9 dim)                                      │    │
│  │    MLP: 9 → 256 → 128 → 8                                        │    │
│  │    Output: extrin_gt (8 dim) + tanh                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 2. Proprio History Encoder (adapt_tconv) - Stage 2 only         │    │
│  │    Input: proprio_hist (50 timesteps × 32 features)              │    │
│  │                                                                   │    │
│  │    Channel Transform:                                             │    │
│  │      Linear(32 → 32) + ReLU → Linear(32 → 32) + ReLU             │    │
│  │                                                                   │    │
│  │    Temporal Aggregation (1D Conv):                                │    │
│  │      Conv1d(32, 32, k=9, s=2) + ReLU  → (N, 32, 21)              │    │
│  │      Conv1d(32, 32, k=5, s=1) + ReLU  → (N, 32, 17)              │    │
│  │      Conv1d(32, 32, k=5, s=1) + ReLU  → (N, 32, 3)               │    │
│  │                                                                   │    │
│  │    low_dim_proj: Linear(32×3=96 → 8)                             │    │
│  │    Output: extrin (8 dim) + tanh                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 3. Actor MLP (actor_mlp)                                         │    │
│  │    Input: obs + extrin (obs_dim + 8)                             │    │
│  │    MLP: (obs_dim+8) → 512 → 256 → 128                            │    │
│  │    Output: state (128 dim)                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │ Value Head (Critic)  │    │ Mu Head (Actor)      │                   │
│  │ Linear(128 → 1)      │    │ Linear(128 → 16)     │                   │
│  │ Output: value        │    │ Output: action mean  │                   │
│  └──────────────────────┘    └──────────────────────┘                   │
│                                                                          │
│  ┌──────────────────────┐                                               │
│  │ Sigma (learnable)    │                                               │
│  │ Parameter(16)        │                                               │
│  │ Output: action std   │                                               │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage별 사용 방식

| Stage | 사용 모듈 | 설명 |
|-------|----------|------|
| **Stage 1 (PPO)** | `env_mlp` + `actor_mlp` | Privileged info를 직접 사용해 policy 학습 |
| **Stage 2 (ProprioAdapt)** | `adapt_tconv` + `actor_mlp` | proprio history로 환경 정보 추론, `env_mlp`은 teacher로만 사용 |

### Config 값 (기본값)
- `actor_units`: [512, 256, 128]
- `priv_mlp_units`: [256, 128, 8]
- `priv_info_dim`: 9
- `actions_num`: 16 (Allegro Hand)
