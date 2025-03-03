import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected, add_self_loops
import torch_geometric as geom
from typing import List, Callable, Optional, Dict
from tqdm import tqdm


class JetDataset(Dataset):
    """Dataset class for jet physics data, creating graph representations for GNN processing."""
    
    # Static PID mapping for all instances
    PID2FLOAT_MAP = {
        22: 0,
        211: .1, -211: .2,
        321: .3, -321: .4,
        130: .5,
        2112: .6, -2112: .7,
        2212: .8, -2212: .9,
        11: 1.0, -11: 1.1,
        13: 1.2, -13: 1.3,
        0: 0,
    }
    
    def __init__(
        self, 
        root: str, 
        filename: str, 
        stop: Optional[int] = None, 
        test: bool = False,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None,
        edge_strategy: str = "physics",
        self_loops: bool = True,
        normalize_features: bool = True
    ):
        """
        Initialize the JetDataset.
        
        Args:
            root: Root directory where the dataset should be stored
            filename: Name of the data file
            stop: Optional maximum number of samples to use
            test: Whether this is a test dataset
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
            edge_strategy: Strategy for connecting nodes ("physics", "knn", "fully_connected")
            self_loops: Whether to add self-loops to graphs
            normalize_features: Whether to normalize node features
        """
        self.test = test
        self.filename = filename
        self.stop = stop
        self.data = None  # Will be loaded during processing
        self.n_graphs = 0  # Will be set during processing
        self.edge_strategy = edge_strategy
        self.self_loops = self_loops
        self.normalize_features = normalize_features
        super(JetDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        """Return the name of the raw file to use."""
        return [self.filename]

    @property
    def processed_file_names(self) -> List[str]:
        """Return the name of the processed files (one per graph)."""
        # Load data to get number of graphs
        if self.data is None:
            self.data = np.load(self.raw_paths[0], allow_pickle=True)
            self.n_graphs = min(self.stop, self.data['X'].shape[0]) if self.stop else self.data['X'].shape[0]

        prefix = 'data_test_' if self.test else 'data_'
        return [f'{prefix}{i}.pt' for i in range(self.n_graphs)]

    def download(self):
        """Download is not implemented as the data is provided locally."""
        # This method is required by the Dataset parent class
        pass

    def process(self):
        """Process the raw data into graph structures for GNN consumption."""
        self.data = np.load(self.raw_paths[0], allow_pickle=True)
        X = self.data['X']  # Jet features
        y = self.data['y']  # Labels (quark=1, gluon=0)
        
        # Map PID values to float representations
        pids = np.unique(X[:, :, 3].flatten())
        for pid in tqdm(pids, desc="Converting PIDs"):
            if pid in self.PID2FLOAT_MAP:
                np.place(X[:, :, 3], X[:, :, 3] == pid, self.PID2FLOAT_MAP[pid])
            else:
                print(f"Warning: Unknown PID {pid} - setting to 0")
                np.place(X[:, :, 3], X[:, :, 3] == pid, 0)
            
        # Process each jet into a graph
        for i in tqdm(range(self.n_graphs), desc="Creating graphs"):
            jet_data = X[i].copy()  # Copy to prevent modifying original data
            jet_label = y[i]
            
            # Remove zero-padded entries
            mask = ~np.all(jet_data == 0, axis=1)
            if np.sum(mask) == 0:
                # Skip empty jets
                print(f"Warning: Jet {i} has no valid particles")
                continue
                
            _jet_data = jet_data[mask]
        
            # Preprocess jet data
            if self.normalize_features:
                self._preprocess_jet_data(_jet_data)

            # Create graph components
            node_feats = self._get_node_features(_jet_data)
            edge_index = self._get_adjacency_info(_jet_data)
            label = torch.tensor([jet_label], dtype=torch.int64)
            
            # Create and save data object
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                y=label
            )
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            file_prefix = 'data_test_' if self.test else 'data_'
            torch.save(
                data, 
                os.path.join(self.processed_dir, f'{file_prefix}{i}.pt')
            )

    def _preprocess_jet_data(self, jet_data: np.ndarray) -> None:
        """
        Preprocess jet data: center jets and normalize pT.
        
        Args:
            jet_data: Jet data to preprocess (modified in-place)
        """
        # Center jets using weighted average of rapidity and phi
        weights = jet_data[:, 0]
        if np.sum(weights) > 0:  # Avoid division by zero
            yphi_avg = np.average(jet_data[:, 1:3], weights=weights, axis=0)
            jet_data[:, 1:3] -= yphi_avg
        
        # Normalize pT
        pt_sum = np.sum(jet_data[:, 0])
        if pt_sum > 0:  # Avoid division by zero
            jet_data[:, 0] /= pt_sum
        
        # Sort by pT (descending)
        sort_indices = jet_data[:, 0].argsort()[::-1]
        jet_data[:] = jet_data[sort_indices].copy()

    def _get_node_features(self, jet_data: np.ndarray) -> torch.Tensor:
        """
        Extract node features from jet data.
        
        Args:
            jet_data: Preprocessed jet data
            
        Returns:
            Tensor with node features
        """
        return torch.tensor(jet_data, dtype=torch.float)

    def _get_adjacency_info(self, jet_data: np.ndarray) -> torch.Tensor:
        """
        Create edge indices for the graph based on selected strategy.
        
        Args:
            jet_data: Preprocessed jet data
            
        Returns:
            Edge index tensor for the graph
        """
        if self.edge_strategy == "physics":
            # Get particles sorted by different physical properties
            pt_order = jet_data[:, 0].argsort()[::-1]
            rapidity_order = jet_data[:, 1].argsort()
            phi_order = jet_data[:, 2].argsort()
            
            # Connect adjacent particles in each ordering
            in_node = np.concatenate((pt_order[:-1], rapidity_order[:-1], phi_order[:-1]))
            out_node = np.concatenate((pt_order[1:], rapidity_order[1:], phi_order[1:]))
            
            # Create edge indices
            edge_indices = np.stack((in_node, out_node), axis=0)
            
        elif self.edge_strategy == "knn":
            # Use k-nearest neighbors based on delta R
            from sklearn.neighbors import NearestNeighbors
            
            # Calculate delta R between all pairs of particles
            # deltaR = sqrt(deltaY^2 + deltaPhi^2)
            coords = jet_data[:, 1:3]  # y, phi
            
            # Use 5 nearest neighbors (can be adjusted)
            k = min(5, len(coords) - 1)
            if k <= 0:
                # Fall back to fully connected for very small jets
                k = 1
                
            nn = NearestNeighbors(n_neighbors=k+1)  # +1 because a point is its own nearest neighbor
            nn.fit(coords)
            
            # Get indices of nearest neighbors
            distances, indices = nn.kneighbors(coords)
            
            in_node = np.repeat(np.arange(len(coords)), k)
            out_node = indices[:, 1:].flatten()  # Skip the first neighbor (self)
            
            # Create edge indices
            edge_indices = np.stack((in_node, out_node), axis=0)
            
        elif self.edge_strategy == "fully_connected":
            # Create fully connected graph
            n = len(jet_data)
            in_node = np.repeat(np.arange(n), n-1)
            out_node = np.array([j for i in range(n) for j in range(n) if i != j])
            
            # Create edge indices
            edge_indices = np.stack((in_node, out_node), axis=0)
            
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")
        
        # Convert to tensor
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        
        # Make graph undirected
        edge_indices = to_undirected(edge_indices)
        
        # Add self-loops if requested
        if self.self_loops:
            edge_indices = add_self_loops(edge_indices)[0]
        
        return edge_indices

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return self.n_graphs

    def get(self, idx: int) -> Data:
        """Get a specific graph by index."""
        file_prefix = 'data_test_' if self.test else 'data_'
        data = torch.load(os.path.join(self.processed_dir, f'{file_prefix}{idx}.pt'))
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
        
    @property
    def num_node_features(self) -> int:
        """Return the number of features per node."""
        if self.len() == 0:
            raise ValueError("Dataset is empty")
        return self[0].num_node_features
        
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        # Binary classification (quark vs gluon)
        return 2