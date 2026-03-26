# keyboard_control.py
# Returns per-agent action IDs for the mixed-motive env with headings.
# Actions:
#   0: collect
#   1: attack
#   2: use_bait
#   3: move_forward
#   4: turn_left
#   5: turn_right
#   6: change_tool

from __future__ import annotations
from typing import Optional
import pygame

# Action IDs (keep in sync with env)
ACT_COLLECT   = 0
ACT_ATTACK    = 1
ACT_USE_BAIT  = 2
ACT_FWD       = 3
ACT_TL        = 4
ACT_TR        = 5
ACT_CHANGE    = 6

# ----------------------------
# Key layouts per agent
# ----------------------------
# Agent 0 controls (left side / WASD cluster)
AGENT0_KEYMAP = {
    pygame.K_w: ACT_FWD,
    pygame.K_a: ACT_TL,
    pygame.K_d: ACT_TR,

    pygame.K_SPACE: ACT_COLLECT,
    pygame.K_f:     ACT_ATTACK,
    pygame.K_b:     ACT_USE_BAIT,
    pygame.K_c:     ACT_CHANGE,
}

# Agent 1 controls (right side / IJKL cluster)
AGENT1_KEYMAP = {
    pygame.K_i: ACT_FWD,
    pygame.K_j: ACT_TL,
    pygame.K_l: ACT_TR,

    pygame.K_h: ACT_COLLECT,
    pygame.K_k: ACT_ATTACK,
    pygame.K_COMMA: ACT_USE_BAIT,   # ',' key
    pygame.K_PERIOD: ACT_CHANGE,    # '.' key
}

# Add more agents by appending dictionaries here if needed
AGENT_KEYMAPS = [AGENT0_KEYMAP, AGENT1_KEYMAP]


def get_agent_action(events, agent_id: int) -> Optional[int]:
    """
    Inspect pygame events and return a single action id (int 0..6) for the given agent
    when a relevant KEYDOWN occurs. Returns None if no relevant key was pressed for
    this agent in this frame.
    """
    keymap = AGENT_KEYMAPS[agent_id] if agent_id < len(AGENT_KEYMAPS) else AGENT_KEYMAPS[0]

    action: Optional[int] = None
    for event in events:
        if event.type == pygame.QUIT:
            # Let caller handle quitting; do not return an action here
            continue
        if event.type == pygame.KEYDOWN:
            # Immediate, discrete action on keydown
            if event.key in keymap:
                action = keymap[event.key]
                # We return the first relevant keydown we see for determinism
                return action

    return action
