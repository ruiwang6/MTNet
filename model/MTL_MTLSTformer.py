import torch.nn as nn
import torch
from torchinfo import summary


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, hidden_dim)
        # K, V (batch_size, ..., src_length, hidden_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = torch.matmul(query,
                                  key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_score, value)  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = hidden_dim)

        out = self.fc(out)

        return out

class SpatialAttentionConvlutionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.graph_conv_param = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, hidden_dim)
        # K, V (batch_size, ..., src_length, hidden_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = torch.matmul(query,
                                  key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)

        # attention spatial graph
        support = torch.eye(attn_score.shape[2]).to(query.device)
        attn_score = attn_score + support

        # graph convolution layer
        out = torch.matmul(attn_score, value)  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = hidden_dim)

        out = self.graph_conv_param(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, hidden_dim, feed_forward_dim=2048, num_heads=8, dropout=0, ifnode=False,
    ):
        super().__init__()

        if ifnode:
            self.attn = SpatialAttentionConvlutionLayer(hidden_dim, num_heads)
        else:
            self.attn = AttentionLayer(hidden_dim, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)  # temporal B T N D -> B N T D
        # spatial B T N D -> B T N D
        # x: (batch_size, ..., length, hidden_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, hidden_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, hidden_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class CrossTaskAttentionLayer(nn.Module):
    def __init__(
            self, hidden_dim,  feed_forward_dim=2048, num_heads=8, dropout=0
    ):
        super().__init__()

        self.attn = AttentionLayer(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        Q = Q.transpose(1, -2)  # temporal B T N D -> B N T D
        K = K.transpose(1, -2)  # temporal B T N D -> B N T D
        V = V.transpose(1, -2)  # temporal B T N D -> B N T D

        # spatial B T N D -> B T N D
        # x: (batch_size, ..., length, hidden_dim)
        residual = Q
        out = self.attn(Q, K, V)  # (batch_size, ..., length, hidden_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, hidden_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(1, -2)
        return out


class MTLSTformer(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_emb_dim=24,
            tod_emb_dim=24,
            dow_emb_dim=24,
            adaptive_emb_dim=80,
            feed_forward_dim=256,
            cross_feed_forward_dim=64,
            cross_heads=4,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_emb_dim = input_emb_dim
        self.tod_emb_dim = tod_emb_dim
        self.dow_emb_dim = dow_emb_dim
        self.adaptive_emb_dim = adaptive_emb_dim
        self.hidden_dim = (
                input_emb_dim
                + tod_emb_dim
                + dow_emb_dim
                + adaptive_emb_dim
        )
        self.cross_feed_forward_dim = cross_feed_forward_dim
        self.cross_heads = cross_heads
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_fc1 = nn.Linear(input_dim, input_emb_dim)
        self.input_fc2 = nn.Linear(input_dim, input_emb_dim)

        if tod_emb_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_emb_dim)
        if dow_emb_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_emb_dim)
        if adaptive_emb_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_emb_dim))
            )

        self.attn_layers_t1 = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout, False)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s1 = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout, True)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_t2 = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout, False)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s2 = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_dim, feed_forward_dim, num_heads, dropout, True)
                for _ in range(num_layers)
            ]
        )

        self.cross_task_inter1 = CrossTaskAttentionLayer(self.hidden_dim, cross_feed_forward_dim, self.cross_heads, dropout)
        self.cross_task_inter2 = CrossTaskAttentionLayer(self.hidden_dim, cross_feed_forward_dim, self.cross_heads, dropout)

        # ouput projection
        self.output_fc1 = nn.Linear(
            in_steps * self.hidden_dim, out_steps * output_dim
        )
        self.output_fc2 = nn.Linear(
            in_steps * self.hidden_dim, out_steps * output_dim
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_emb_dim > 0:
            tod = x[..., 2]
        if self.dow_emb_dim > 0:
            dow = x[..., 3]

        x1 = x[..., [0, 2, 3]]
        x2 = x[..., [1, 2, 3]]

        x1 = self.input_fc1(x1)  # (batch_size, in_steps, num_nodes, input_emb_dim)
        x2 = self.input_fc2(x2)  # (batch_size, in_steps, num_nodes, input_emb_dim)

        features = []
        if self.tod_emb_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_emb_dim)
            features.append(tod_emb)
        if self.dow_emb_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_emb_dim)
            features.append(dow_emb)
        if self.adaptive_emb_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        features = torch.cat(features, dim=-1)
        x1 = torch.concat([x1, features], dim=-1)  # (batch_size, in_steps, num_nodes, hidden_dim)
        x2 = torch.concat([x2, features], dim=-1)  # (batch_size, in_steps, num_nodes, hidden_dim)

        x1_cross_input = x1
        x2_cross_input = x2

        x1 = self.cross_task_inter1(x1_cross_input, x2_cross_input, x2_cross_input)
        x2 = self.cross_task_inter2(x2_cross_input, x1_cross_input, x1_cross_input)

        # temporal attention for two tasks
        for attn in self.attn_layers_t1:
            x1 = attn(x1, dim=1)  # B T N D
        for attn in self.attn_layers_s1:
            x1 = attn(x1, dim=2)  #
        # (batch_size, in_steps, num_nodes, hidden_dim)

        # spatial attention graph convolutional for two tasks
        for attn in self.attn_layers_t2:
            x2 = attn(x2, dim=1)  # B T N D
        for attn in self.attn_layers_s2:
            x2 = attn(x2, dim=2)  #

        # output projection
        out1 = x1.transpose(1, 2)  # (batch_size, num_nodes, in_steps, hidden_dim)
        out1 = out1.reshape(
            batch_size, self.num_nodes, self.in_steps * self.hidden_dim
        )
        out1 = self.output_fc1(out1).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out1 = out1.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        out2 = x2.transpose(1, 2)  # (batch_size, num_nodes, in_steps, hidden_dim)
        out2 = out2.reshape(
            batch_size, self.num_nodes, self.in_steps * self.hidden_dim
        )
        out2 = self.output_fc2(out2).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out2 = out2.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return torch.cat([out1, out2], dim=-1)

