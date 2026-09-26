from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
import numpy as np


class GACGroupMAC(BasicMAC):
    """
    Multi-Agent Controller for GAC-GoMARL.
    Đồng bộ cấu trúc chia nhóm `group` từ Mixer, tính toán Attention Group Mask
    với sự hỗ trợ của episode_mask (loại bỏ padding/agent kết thúc) và điều phối forward agent.
    """
    def __init__(self, scheme, groups, args):
        super(GACGroupMAC, self).__init__(scheme, groups, args)
        self.group = args.group
        if self.group is None:
            self.group = [[i for i in range(self.n_agents)]]

        self.group_states = None
        self.last_beta_mean = 0.0
        self.last_isolated_ratio = 0.0

    def update_group(self, new_group):
        """Cập nhật cấu trúc nhóm mới nhất từ Mixer/Learner."""
        self.group = new_group

    def build_group_mask(self, bs, group_structure, device, episode_mask=None):
        """
        Tạo ma trận group_mask (bs, n_agents, n_agents):
        - mask[b, i, j] = 1.0 nếu agent i và j cùng nhóm và i != j.
        - Kết hợp với episode_mask (nếu có) để loại bỏ agents bị padding.
        """
        mask = th.zeros((bs, self.n_agents, self.n_agents), device=device)
        for g in group_structure:
            for i in g:
                for j in g:
                    if i != j:
                        mask[:, i, j] = 1.0

        if episode_mask is not None:
            # episode_mask shape: (bs, n_agents)
            valid = episode_mask.unsqueeze(1) * episode_mask.unsqueeze(2)
            mask = mask * valid

        return mask

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        bs = ep_batch.batch_size
        device = agent_inputs.device

        # Lấy episode mask nếu có trong batch (filled hoặc 1 - terminated)
        episode_mask = None
        if "filled" in ep_batch.scheme:
            # filled shape: (bs, max_t, 1) -> expand to (bs, n_agents)
            filled = ep_batch["filled"][:, t].squeeze(-1).float()
            episode_mask = filled.unsqueeze(-1).expand(bs, self.n_agents)

        # Xây dựng group mask động theo cấu trúc nhóm hiện hành
        group_mask = self.build_group_mask(bs, self.group, device, episode_mask=episode_mask)

        # Forward qua GACGroupAgent
        out = self.agent(
            inputs=agent_inputs,
            hidden_state=self.hidden_states,
            group_mask=group_mask,
            episode_mask=episode_mask
        )

        agent_outs = out["q_vals"]
        self.hidden_states = out["hidden_states"]
        self.group_states = out["group_states"]  # agent info e^a
        self.w1 = out["w1"]
        self.last_beta_mean = out["beta_mean"].item() if hasattr(out["beta_mean"], "item") else float(out["beta_mean"])
        self.last_isolated_ratio = out["isolated_ratio"].item() if hasattr(out["isolated_ratio"], "item") else float(out["isolated_ratio"])

        return agent_outs
