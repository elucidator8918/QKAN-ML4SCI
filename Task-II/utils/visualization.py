import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch_geometric.data as geom_data
import networkx as nx
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import itertools
import json


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16


def plot_training_curves(
    log_dir: str, 
    version: str, 
    metrics: List[str] = None, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves from logged metrics.
    
    Args:
        log_dir: Directory containing logs
        version: Version of the run to plot
        metrics: List of metrics to plot (default: loss and auc)
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Load CSV file
    try:
        csv_path = os.path.join(log_dir, "lightning_logs", version, "metrics.csv")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        csv_path = os.path.join(log_dir, version, "metrics.csv")
        df = pd.read_csv(csv_path)
    
    # Group by epoch
    df_grouped = df.groupby("epoch").mean()
    
    # Default metrics
    if metrics is None:
        metrics = [
            ("loss", "Loss Curves"),
            ("auc", "AUC ROC Scores")
        ]
    
    # Create figure with subplots
    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    # Handle single metric case
    if num_plots == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        
        # Find all columns containing the metric name
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        
        if train_col in df_grouped.columns and val_col in df_grouped.columns:
            ax.plot(df_grouped.index, df_grouped[train_col], label="Training")
            ax.plot(df_grouped.index, df_grouped[val_col], label="Validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Metric '{metric}' not found", 
                    ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_models(
    log_dirs: List[str], 
    versions: List[str], 
    model_names: List[str],
    metrics: List[str] = None, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare training results from multiple models.
    
    Args:
        log_dirs: List of log directories
        versions: List of versions to compare
        model_names: Names to use for each model in the plot
        metrics: List of metrics to plot (default: val_auc and val_loss)
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Validate inputs
    if len(log_dirs) != len(versions) or len(versions) != len(model_names):
        raise ValueError("log_dirs, versions, and model_names must have the same length")
    
    # Set plotting style
    set_plotting_style()
    
    # Default metrics
    if metrics is None:
        metrics = ["val_auc", "val_loss"]
    
    # Create figure with subplots
    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    # Handle single metric case
    if num_plots == 1:
        axes = [axes]
    
    # Plot metrics for all models
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for log_dir, version, name in zip(log_dirs, versions, model_names):
            try:
                csv_path = os.path.join(log_dir, "lightning_logs", version, "metrics.csv")
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                csv_path = os.path.join(log_dir, version, "metrics.csv")
                df = pd.read_csv(csv_path)
            
            df_grouped = df.groupby("epoch").mean()
            
            if metric in df_grouped.columns:
                ax.plot(df_grouped.index, df_grouped[metric], label=name)
                ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric.split('_')[-1].upper())
                ax.legend()
            else:
                print(f"Metric '{metric}' not found in logs for {name}")
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (thresholded if necessary)
        class_names: Names of classes
        normalize: Whether to normalize by row
        title: Title for the plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Default class names
    if class_names is None:
        class_names = ["Gluon", "Quark"]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores (before thresholding)
        title: Title for the plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Configure axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores (before thresholding)
        title: Title for the plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    
    # Configure axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_graph(
    data: geom_data.Data,
    title: str = "Graph Visualization",
    color_by: str = "node_features",
    feature_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize a graph from PyG data.
    
    Args:
        data: PyG data object
        title: Title for the plot
        color_by: How to color nodes ('node_features', 'degree', or 'centrality')
        feature_idx: Index of node feature to use for coloring (if color_by='node_features')
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Convert to networkx graph
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Prepare node colors
    if color_by == 'node_features' and hasattr(data, 'x'):
        if feature_idx < data.x.shape[1]:
            node_colors = data.x[:, feature_idx].cpu().numpy()
        else:
            node_colors = np.ones(data.x.shape[0])
    elif color_by == 'degree':
        node_colors = [d for _, d in G.degree()]
    elif color_by == 'centrality':
        node_colors = list(nx.betweenness_centrality(G).values())
    else:
        node_colors = 'skyblue'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw graph
    nx.draw_networkx(
        G, 
        pos=pos,
        with_labels=False,
        node_color=node_colors,
        node_size=50,
        edge_color='gray',
        alpha=0.8,
        width=0.5,
        cmap=plt.cm.viridis,
        ax=ax
    )
    
    # Add colorbar if using node colors
    if color_by in ['node_features', 'degree', 'centrality']:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{color_by.replace("_", " ").title()}')
    
    # Configure plot
    ax.set_title(title)
    ax.axis('off')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Node Embeddings (t-SNE)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Node or graph embeddings
        labels: Labels for coloring
        title: Title for the plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot embeddings
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='coolwarm',
        alpha=0.8,
        s=50
    )
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)
    
    # Configure plot
    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    model, 
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        title: Title for the plot
        top_n: Number of top features to show
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Set plotting style
    set_plotting_style()
    
    # Extract feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Try to extract from model weights
        try:
            # Get first layer weights
            weights = model.model.gnn.layers[0].conv.weight.detach().cpu().numpy()
            importances = np.mean(np.abs(weights), axis=0)
        except (AttributeError, IndexError):
            raise ValueError("Model does not support feature importance extraction")
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(top_indices))
    ax.barh(y_pos, importances[top_indices], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_model_summary(
    model, 
    results: Dict, 
    hyperparams: Dict,
    filename: str = "model_summary.json"
) -> None:
    """
    Save model summary, results, and hyperparameters to JSON.
    
    Args:
        model: Trained model
        results: Dictionary of evaluation results
        hyperparams: Dictionary of hyperparameters
        filename: Output filename
    """
    # Create summary dictionary
    summary = {
        "model_type": model.__class__.__name__,
        "hyperparameters": hyperparams,
        "results": results,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return str(obj)
    
    # Make all values JSON serializable
    summary_serializable = json.loads(
        json.dumps(summary, default=make_serializable)
    )
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(summary_serializable, f, indent=2)