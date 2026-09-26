import torch
import torch.nn as nn
import torch.nn.functional as F


class GACGroupAgent(nn.Module):
    """
    Group-Aware Communication Agent for GoMARL (GAC-GoMARL).
    Kế thừa kiến trúc của Specialized Agent Network (SAN) trong GoMARL:
    GRU -> Group-Restricted Attention & Gating -> Encoder fe (sinh agent info e^a) -> Decoder fd (sinh W_Q, b_Q).
    Đồng thời giữ nhánh w1_generator độc lập từ hidden state gốc h để định nghĩa topology chia nhóm.
    """

    def __init__(self, input_shape, args):
        super(GACGroupAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.rnn_hidden_dim
        self.comm_dim = getattr(args, "gac_comm_dim", 32)
        self.use_gate = getattr(args, "gac_use_gate", True)
        self.explicit_comm = getattr(args, "gac_explicit_comm", False)
        self.carry_fused_hidden = getattr(args, "gac_carry_fused_hidden", False)

        # ---- Phân nhánh GỐC của GoMARL ----
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # w1 generator: Dùng h gốc để sinh trọng số cá nhân phục vụ Automatic Grouping.
        # Nhánh này KHÔNG đi qua comm module -> Không bị chi phối ngược bởi communication.
        self.w1_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, args.hypernet_embed),
            nn.Tanh()
        )

        # Encoder fe: sinh agent info e^a (kích thước hypernet_embed) từ h_fused
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, args.hypernet_embed),
            nn.Tanh()
        )

        # Decoder fd: sinh W_Q và b_Q từ agent info e^a
        self.decoder_w = nn.Sequential(
            nn.Linear(args.hypernet_embed, self.hidden_dim * args.n_actions)
        )
        self.decoder_b = nn.Sequential(
            nn.Linear(args.hypernet_embed, args.n_actions)
        )

        # ---- Phân nhánh MỚI: Group-Aware Communication (GAC) ----
        self.W_q = nn.Linear(self.hidden_dim, self.comm_dim, bias=False)
        self.W_k = nn.Linear(self.hidden_dim, self.comm_dim, bias=False)
        
        if self.explicit_comm:
            self.explicit_feat_dim = getattr(args, "explicit_comm_feat_dim", 10)
            self.W_v = nn.Linear(self.explicit_feat_dim, self.comm_dim, bias=False)
        else:
            self.W_v = nn.Linear(self.hidden_dim, self.comm_dim, bias=False)

        # Chiếu context message về kích thước hidden_dim (bias=False để vector 0 chiếu thành 0)
        self.comm_proj = nn.Linear(self.comm_dim, self.hidden_dim, bias=False)

        if self.use_gate:
            self.gate_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Auxiliary loss decoder (tuỳ chọn ablation)
        self.aux_loss_weight = getattr(args, "aux_comm_loss_weight", 0.0)
        if self.aux_loss_weight > 0:
            obs_shape = input_shape
            self.aux_decoder = nn.Linear(self.comm_dim, obs_shape)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, group_mask=None, explicit_feats=None, episode_mask=None):
        """
        inputs        : (bs, n_agents, input_shape) hoặc (bs*n_agents, input_shape)
        hidden_state  : (bs, n_agents, hidden_dim) hoặc (bs*n_agents, hidden_dim)
        group_mask    : (bs, n_agents, n_agents) — 1 nếu cùng nhóm & không phải self-loop
        explicit_feats: (bs, n_agents, explicit_dim) hoặc (bs*n_agents, explicit_dim)
        episode_mask  : (bs, n_agents) — 1 nếu agent còn hợp lệ trong episode
        """
        if inputs.dim() == 3:
            bs, n, feat_dim = inputs.shape
        else:
            n = self.n_agents
            bs = inputs.shape[0] // n
            feat_dim = inputs.shape[-1]

        inputs = inputs.reshape(bs * n, feat_dim)
        hidden_state = hidden_state.reshape(bs * n, self.hidden_dim)

        # 1. GRU Step
        x = F.relu(self.fc1(inputs), inplace=True)
        h = self.rnn(x, hidden_state)  # (bs * n, hidden_dim)

        # 2. w1 Generator (sử dụng h gốc, độc lập hoàn toàn với GAC)
        # Giữ tính nhất quán: w1 phục vụ cho dynamic grouping module
        w1 = self.w1_generator(h.detach() if getattr(self.args, "detach_w1", True) else h)  # (bs * n, embed_dim)

        # 3. Group-Restricted Attention
        if group_mask is None:
            # Mặc định tất cả chung 1 nhóm nếu không truyền mask
            group_mask = torch.ones((bs, n, n), device=inputs.device) - torch.eye(n, device=inputs.device).unsqueeze(0)

        h_reshaped = h.view(bs, n, self.hidden_dim)
        q = self.W_q(h_reshaped)  # (bs, n, comm_dim)
        k = self.W_k(h_reshaped)  # (bs, n, comm_dim)

        if self.explicit_comm and explicit_feats is not None:
            v_in = explicit_feats.view(bs, n, -1)
            v = self.W_v(v_in)
        else:
            v = self.W_v(h_reshaped)  # (bs, n, comm_dim)

        # Raw attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.comm_dim ** 0.5)  # (bs, n, n)

        # Áp dụng group_mask và episode_mask
        attn_mask = group_mask.clone()
        if episode_mask is not None:
            valid = episode_mask.unsqueeze(1) * episode_mask.unsqueeze(2)  # (bs, n, n)
            attn_mask = attn_mask * valid

        scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Xử lý các agent bị cô lập (|g_j| = 1 hoặc hàng toàn mask 0)
        row_has_neighbor = (attn_mask.sum(dim=-1, keepdim=True) > 0)  # (bs, n, 1)
        alpha = F.softmax(scores, dim=-1)
        alpha = alpha * row_has_neighbor.float()  # Ép toàn bộ hàng cô lập thành 0

        # Context message
        c = torch.bmm(alpha, v)  # (bs, n, comm_dim)
        c = self.comm_proj(c).view(bs * n, self.hidden_dim)  # (bs * n, hidden_dim)

        # 4. Adaptive Gating Fusion
        if self.use_gate:
            beta = torch.sigmoid(self.gate_fc(torch.cat([h, c], dim=-1)))
            h_fused = h + beta * c
            beta_mean = beta.mean()
        else:
            h_fused = h + c
            beta_mean = torch.tensor(0.0, device=inputs.device)

        # 5. SAN Encoder & Decoder (Dùng h_fused trực tiếp, KHÔNG detach để truyền gradient thông suốt)
        e = self.encoder(h_fused)  # agent info e^a, shape (bs*n, embed_dim)

        fc2_w = self.decoder_w(e).reshape(bs * n, self.hidden_dim, self.args.n_actions)
        fc2_b = self.decoder_b(e).reshape(bs * n, 1, self.args.n_actions)

        # Q^a = h_fused · W_Q + b_Q
        h_fused_in = h_fused.reshape(bs * n, 1, self.hidden_dim)
        q_vals = torch.matmul(h_fused_in, fc2_w) + fc2_b
        q_vals = q_vals.reshape(bs, n, -1)

        # Quyết định next hidden state (mặc định giữ h gốc để tránh nhiễu tích luỹ)
        next_h = h_fused if self.carry_fused_hidden else h
        next_h = next_h.reshape(bs, n, -1)

        # Reshape agent info e^a thành (bs, n, embed_dim)
        agent_info_e = e.reshape(bs, n, -1)

        # Tỷ lệ agent bị cô lập (|g_j| = 1)
        isolated_ratio = (row_has_neighbor.float() == 0).float().mean()

        return {
            "q_vals": q_vals,
            "hidden_states": next_h,
            "group_states": agent_info_e,  # agent info e^a đưa vào mixer / SD loss
            "w1": w1.reshape(bs, n, -1),
            "beta_mean": beta_mean,
            "isolated_ratio": isolated_ratio,
            "c": c.reshape(bs, n, -1)
        }
