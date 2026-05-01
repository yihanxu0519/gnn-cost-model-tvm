
import os
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool


@dataclass
class HParams:
    input_dim: int = 16
    hidden_dim: int = 64
    global_feat_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.1

    lr: float = 1e-3
    wd: float = 1e-4
    bs: int = 64
    max_epochs: int = 60
    grad_clip: float = 1.0

    # ranking loss params 
    rank_w: float = 1.0
    rank_margin: float = 0.005

    patience: int = 15
    min_delta: float = 1e-4

    
    log_clip: float = 30.0

    save_dir: str = "outputs"
    ckpt_name: str = "gnn_model_v2.pth"

    @property
    def save_path(self):
        return os.path.join(self.save_dir, self.ckpt_name)


# 
# Model
# 

class CostModel(nn.Module):



    def __init__(self, in_dim=16, hid=64, n_layers=3,
                 drop=0.1, gf_dim=32):
        super().__init__()
        self.gf_dim = gf_dim

        # project raw 16d node features into hidden space
        self.node_proj = nn.Linear(in_dim, hid)

        # graph conv stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SAGEConv(hid, hid))
            self.norms.append(nn.LayerNorm(hid))
        self.drop = nn.Dropout(drop)

        # global feature branch — small MLP
        # 32 -> 64 -> 32, with layernorm at input
        self.gf_branch = nn.Sequential(
            nn.LayerNorm(gf_dim),
            nn.Linear(gf_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
        )

        # regression head: takes concat of graph_emb (64d)
        # and gf_out (32d) = 96d total
        cat_dim = hid + hid // 2
        self.regressor = nn.Sequential(
            nn.Linear(cat_dim, hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid // 2, 1),
        )

    def forward(self, data):
        x = data.x
        ei = data.edge_index
        batch = data.batch

        # node embedding
        x = F.relu(self.node_proj(x))

        # message passing with residual
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, ei)
            x = norm(x)
            x = F.relu(x)
            x = self.drop(x)
            x = x + residual

        # pool nodes -> one vector per graph
        g_emb = global_mean_pool(x, batch)

        # handle the global features
       
        gf = getattr(data, "global_feat", None)
        if gf is not None:
            if gf.dim() == 3:
                gf = gf.squeeze(1)
            gf = gf.float()
        else:
            gf = torch.zeros(g_emb.size(0), self.gf_dim,
                             device=g_emb.device)

        gf_out = self.gf_branch(gf)

        combined = torch.cat([g_emb, gf_out], dim=-1)
        out = self.regressor(combined).squeeze(-1)
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# 
# Loss functions
#
#



def ranking_loss(pred, target, wl_ids, margin=0.005):

    if pred.numel() < 2:
        return pred.new_tensor(0.0)

   
    same_wl = (wl_ids.unsqueeze(1) == wl_ids.unsqueeze(0))

    t_diff = target.unsqueeze(1) - target.unsqueeze(0)
    p_diff = pred.unsqueeze(1) - pred.unsqueeze(0)

    # only look at pairs with meaningful difference
    valid = same_wl & (t_diff.abs() > margin)

    if valid.sum() == 0:
        return pred.new_tensor(0.0)

    sign = torch.sign(t_diff)
    losses = F.relu(margin - sign * p_diff)
    return losses[valid].mean()


# Metrics


def _rank_array(arr):
  
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    return ranks


def spearman(pred, target):
    if len(pred) < 2:
        return 0.0
    rp = _rank_array(pred)
    rt = _rank_array(target)
    # standardise
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    rt = (rt - rt.mean()) / (rt.std() + 1e-12)
    return float((rp * rt).mean())


def per_workload_spearman(preds, targets, wl_ids):

    results = []
    for wid in np.unique(wl_ids):
        mask = (wl_ids == wid)
        n = mask.sum()
        if n < 5:
            # not enough samples to get a meaningful correlation
            continue
        rho = spearman(preds[mask], targets[mask])
        results.append(rho)

    if len(results) == 0:
        return 0.0
    return float(np.mean(results))



# Data loading


def load_graphs(data_dir):
    pt_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".pt"):
                pt_files.append(os.path.join(root, f))
    pt_files.sort()

    graphs = []
    n_bad = 0
    for fpath in pt_files:
        try:
            g = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception:
            n_bad += 1
            continue

        # sanity check 
        if not hasattr(g, "x") or not hasattr(g, "edge_index") or not hasattr(g, "y"):
            n_bad += 1
            continue

        # normalise y to tensor
        if isinstance(g.y, torch.Tensor):
            g.y = g.y.view(1).float()
        else:
            g.y = torch.tensor([float(g.y)], dtype=torch.float32)

        
        wl_name = None
        if hasattr(g, "meta") and isinstance(g.meta, dict):
            wl_name = g.meta.get("prefix", None)
        if wl_name is None:
            wl_name = os.path.basename(os.path.dirname(fpath))
            # if the parent dir is the root dir itself, just call it unknown
            if wl_name == os.path.basename(data_dir):
                wl_name = "unknown"

        g._wl_name = str(wl_name)
        graphs.append(g)

    print(f"Loaded {len(graphs)} graphs ({n_bad} skipped)")
    return graphs


def assign_wl_ids(graphs):
  
    names = sorted(set(g._wl_name for g in graphs))
    name2id = {name: i for i, name in enumerate(names)}
    for g in graphs:
        g.workload_id = torch.tensor([name2id[g._wl_name]], dtype=torch.long)
    return name2id


def make_splits(graphs, seed=0, ratios=(0.7, 0.15, 0.15)):

    train_r, val_r, _ = ratios

    # group by workload
    buckets = defaultdict(list)
    for i, g in enumerate(graphs):
        buckets[g._wl_name].append(i)

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []

    for wl_name in sorted(buckets.keys()):
        idxs = np.array(buckets[wl_name])
        rng.shuffle(idxs)
        n = len(idxs)

        # if too few samples just dump them all in train
        if n < 5:
            tr.extend(idxs.tolist())
            continue

        n_tr = int(n * train_r)
        n_va = int(n * val_r)

        tr.extend(idxs[:n_tr].tolist())
        va.extend(idxs[n_tr:n_tr + n_va].tolist())
        te.extend(idxs[n_tr + n_va:].tolist())

    return ([graphs[i] for i in tr],
            [graphs[i] for i in va],
            [graphs[i] for i in te])



# Training loop


def run_one_epoch(model, loader, optimizer, hp, is_train):
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0
    all_pred, all_tgt, all_wl = [], [], []

    reg_fn = nn.SmoothL1Loss()

    for batch in loader:
        batch = batch.to(next(model.parameters()).device)

        pred = model(batch)
        tgt = batch.y.view(-1).float()
        wids = batch.workload_id.view(-1)

        # clamp to avoid numerical issues on weird outliers
        pred_c = torch.clamp(pred, -hp.log_clip, hp.log_clip)
        tgt_c = torch.clamp(tgt, -hp.log_clip, hp.log_clip)

        loss = reg_fn(pred_c, tgt_c)
        loss = loss + hp.rank_w * ranking_loss(
            pred_c, tgt_c, wids, hp.rank_margin
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()

        bs = tgt.numel()
        total_loss += loss.item() * bs
        n_samples += bs

        all_pred.append(pred.detach().cpu())
        all_tgt.append(tgt.detach().cpu())
        all_wl.append(wids.detach().cpu())

    # compute metrics
    preds_np = torch.cat(all_pred).numpy()
    tgts_np = torch.cat(all_tgt).numpy()
    wls_np = torch.cat(all_wl).numpy()

    avg_loss = total_loss / max(n_samples, 1)
    sp = per_workload_spearman(preds_np, tgts_np, wls_np)

    return {"loss": avg_loss, "spearman": sp}


def train(model, train_loader, val_loader, hp, device):
     opt = torch.optim.AdamW(model.parameters(), lr=hp.lr,
                            weight_decay=hp.wd)

    best_sp = -1.0
    best_epoch = -1
    wait = 0

    for epoch in range(hp.max_epochs):
        tr = run_one_epoch(model, train_loader, opt, hp, True)
        val = run_one_epoch(model, val_loader, None, hp, False)

        print(f"[{epoch+1:3d}/{hp.max_epochs}] "
              f"train_loss={tr['loss']:.4f}  "
              f"val_loss={val['loss']:.4f}  "
              f"val_spearman={val['spearman']:.4f}")

        # check if we improved enough
        improved = (val["spearman"] - best_sp) > hp.min_delta
        if improved:
            best_sp = val["spearman"]
            best_epoch = epoch
            wait = 0
            save_checkpoint(model, hp, epoch, best_sp)
            print(f"  -> saved (sp={best_sp:.4f})")
        else:
            wait += 1
            if wait >= hp.patience:
                print(f"Stopping early at epoch {epoch+1}. "
                      f"Best was epoch {best_epoch+1} "
                      f"(sp={best_sp:.4f})")
                break

    return best_sp, best_epoch


def save_checkpoint(model, hp, epoch, sp):
    os.makedirs(hp.save_dir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "hparams": hp.__dict__,
        "best_spearman": sp,
    }, hp.save_path)





def main():
    hp = HParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    all_graphs = load_graphs("dataset_sched")
    wl_map = assign_wl_ids(all_graphs)
    print(f"Workloads found: {list(wl_map.keys())}")

    train_set, val_set, test_set = make_splits(all_graphs, seed=0)
    print(f"Split sizes: train={len(train_set)}, "
          f"val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=hp.bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hp.bs)
    test_loader = DataLoader(test_set, batch_size=hp.bs)

    # build model
    model = CostModel(
        in_dim=hp.input_dim,
        hid=hp.hidden_dim,
        n_layers=hp.num_layers,
        drop=hp.dropout,
        gf_dim=hp.global_feat_dim,
    ).to(device)
    print(f"Model has {model.count_params():,} parameters")

    # train
    train(model, train_loader, val_loader, hp, device)

    # load best checkpoint and evaluate on test
    ckpt = torch.load(hp.save_path, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_res = run_one_epoch(model, val_loader, None, hp, False)
    test_res = run_one_epoch(model, test_loader, None, hp, False)

    print()
    print("=" * 50)
    print("Final results (best checkpoint):")
    print(f"  Val  — loss={val_res['loss']:.4f}, "
          f"spearman={val_res['spearman']:.4f}")
    print(f"  Test — loss={test_res['loss']:.4f}, "
          f"spearman={test_res['spearman']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()