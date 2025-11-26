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
