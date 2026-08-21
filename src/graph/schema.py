import pandas as pd
import dgl
import torch
import os

def build_heterograph(transactions_df: pd.DataFrame):
    """
    Builds a heterogeneous graph from transaction logs.
    Edges based on spec:
    (user, transacts, transaction)
    (transaction, initiated_by, user)
    (transaction, paid_via, instrument)
    (user, owns, device)
    """
    # Create unique mappings
    users = transactions_df['user_id'].unique()
    txs = transactions_df['transaction_id'].unique()
    devices = transactions_df['device_hash'].unique()
    instruments = transactions_df['instrument_id'].unique()
    
    u_map = {u: i for i, u in enumerate(users)}
    t_map = {t: i for i, t in enumerate(txs)}
    d_map = {d: i for i, d in enumerate(devices)}
    i_map = {i: idx for idx, i in enumerate(instruments)}
    
    # Map edges
    u_idx = transactions_df['user_id'].map(u_map).values
    t_idx = transactions_df['transaction_id'].map(t_map).values
    d_idx = transactions_df['device_hash'].map(d_map).values
    i_idx = transactions_df['instrument_id'].map(i_map).values
    
    # DGL expects (source_nodes, dest_nodes) as tensors
    graph_data = {
        ('user', 'transacts', 'transaction'): (torch.tensor(u_idx), torch.tensor(t_idx)),
        ('transaction', 'initiated_by', 'user'): (torch.tensor(t_idx), torch.tensor(u_idx)),
        
        ('user', 'owns', 'device'): (torch.tensor(u_idx), torch.tensor(d_idx)),
        ('device', 'owned_by', 'user'): (torch.tensor(d_idx), torch.tensor(u_idx)),
        
        ('transaction', 'paid_via', 'instrument'): (torch.tensor(t_idx), torch.tensor(i_idx)),
        ('instrument', 'used_in', 'transaction'): (torch.tensor(i_idx), torch.tensor(t_idx)),
        
        ('user', 'uses_instrument', 'instrument'): (torch.tensor(u_idx), torch.tensor(i_idx)),
    }
    
    g = dgl.heterograph(graph_data)
    
    # Add dummy node features (e.g., degree or just ones for the R-GCN to start with)
    for ntype in g.ntypes:
        g.nodes[ntype].data['feat'] = torch.ones(g.num_nodes(ntype), 10)
        
    return g, users
