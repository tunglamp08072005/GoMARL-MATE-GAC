REGISTRY = {}

from .n_group_agent import GroupAgent
from .gac_group_agent import GACGroupAgent

REGISTRY["n_group"] = GroupAgent
REGISTRY["gac_group"] = GACGroupAgent
