from hora.tasks.allegro_hand_hora import AllegroHandHora
from hora.tasks.allegro_hand_grasp import AllegroHandGrasp

# Mappings from strings to environments
# Map task names (strings) to their environment classes.
# Multiple task names can point to the same class when the implementation is identical
# but differs only in configuration (e.g., left vs. right hand or other config parameters).
isaacgym_task_map = {
    # Right hand
    "RightCorlAllegroHandHora": AllegroHandHora,
    "RightCorlAllegroHandGrasp": AllegroHandGrasp,
    "RightAllegroHandHora": AllegroHandHora,
    "RightAllegroHandGrasp": AllegroHandGrasp,
    "RightTipAllegroHandHora": AllegroHandHora,
    "RightTipAllegroHandGrasp": AllegroHandGrasp,
    # Left hand
    "LeftCorlAllegroHandHora": AllegroHandHora,
    "LeftCorlAllegroHandGrasp": AllegroHandGrasp,
    "LeftAllegroHandHora": AllegroHandHora,
    "LeftAllegroHandGrasp": AllegroHandGrasp,
    "LeftTipAllegroHandHora": AllegroHandHora,
    "LeftTipAllegroHandGrasp": AllegroHandGrasp,
}
