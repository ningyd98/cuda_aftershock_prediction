import torch
from src.models_gnn import STGNNPredictor

B = 2
N = 10
D = 5

model = STGNNPredictor(event_feature_dim=D, global_feature_dim=3)
model = model.cuda()

seq_x = torch.randn(B, N, D).cuda()
global_x = torch.randn(B, 3).cuda()
seq_padding_mask = torch.zeros(B, N, dtype=torch.bool).cuda()
graph_coords_km = torch.randn(B, N, 2).cuda()
graph_strike_rad = torch.randn(B) # CPU tensor

try:
    model(seq_x, global_x, seq_padding_mask, graph_coords_km=graph_coords_km, graph_strike_rad=graph_strike_rad)
except Exception as e:
    import traceback
    traceback.print_exc()
