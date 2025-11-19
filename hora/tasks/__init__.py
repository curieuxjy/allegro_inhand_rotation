from hora.tasks.allegro_hand_hora import AllegroHandHora
from hora.tasks.allegro_hand_grasp import AllegroHandGrasp

# Mappings from strings to environments
# Map task names (strings) to their environment classes.
# Multiple task names can point to the same class when the implementation is identical
# but differs only in configuration (e.g., left vs. right hand or other config parameters).
isaacgym_task_map = {
    "AllegroHandHora": AllegroHandHora,           # hora original
    "AllegroHandGrasp": AllegroHandGrasp,
    "RightAllegroHandHora": AllegroHandHora,      # allgero v4 right hand
    "RightAllegroHandGrasp": AllegroHandGrasp,
    "LeftAllegroHandHora": AllegroHandHora,       # allgero v4 left hand
    "LeftAllegroHandGrasp": AllegroHandGrasp,
}
