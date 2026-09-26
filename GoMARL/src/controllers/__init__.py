REGISTRY = {}

from .basic_controller import BasicMAC
from .group_controller import NMAC as GroupMAC
from .gac_group_controller import GACGroupMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["group_mac"] = GroupMAC
REGISTRY["gac_group_mac"] = GACGroupMAC
