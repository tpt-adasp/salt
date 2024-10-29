import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patheffects import withStroke
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout

from . import general_utils as utils

class MappingExplorer():
  """MappingExplorer class with tools to explore the mapping of the
  taxonomy
  """
  def __init__(self, map_md_file_path, roots_md_file_path, std_col):

    assert os.path.isfile(map_md_file_path)

    self._map_md_file_path = map_md_file_path
    self._roots_md_file_path = roots_md_file_path
    self._std_col = std_col

    self._init_map_df()
    self._init_roots_dict()
    self._init_leaf()

  ### --- Init functions

  def _init_map_df(self):
    """Load mapping from csv file in a dataframe
    """
    map_df = pd.read_csv(self._map_md_file_path, sep='\t')

    # Discard all empty rows
    map_df = map_df.loc[map_df[self._std_col].notna()].reset_index(drop=True)

    # Replace nan values with empty strings
    map_df.replace(pd.NA, '', inplace=True)

    self.map_df = map_df


  def _init_roots_dict(self):
    """Read taxonomy's root dictionaries from json file

    Raises
    ------
    ValueError
        If a standardized root label is not found in the mapping dataframe
    """
    if os.path.isfile(self._roots_md_file_path):
      with open(self._roots_md_file_path, 'r', encoding='utf-8') as json_file:
        self.roots = json.load(json_file)
    else:
      self.generate_taxonomy_roots()

    for root in self.roots.keys():
      if root not in self.map_df[self._std_col].unique():
        raise ValueError('Not all root labels were found in the mapping file. '
                         'Perhaps the roots_md_file_path is incorect...')

  def _init_leaf(self):
    def get_empty_list_keys(d):
      empty_keys = []
      for k, v in d.items():
        if isinstance(v, list) and not v:  # If the value is an empty list
          empty_keys.append(k)
        elif isinstance(v, dict):  # If the value is a dictionary, recurse into it
          nested_empty_keys = get_empty_list_keys(v)
          if not nested_empty_keys and not v:  # If nested dictionary is empty
            empty_keys.append(k)
          else:
            empty_keys.extend(nested_empty_keys)
      return list(set(empty_keys))

    self.leaf_nodes = get_empty_list_keys(self.roots)


  def generate_taxonomy_roots(self):
    """Generates a json file with tree-like dictionary for each standardized
    root label
    """
    if os.path.exists(self._roots_md_file_path):
      overwrite = utils.confirm_overwriting(self._roots_md_file_path,
                                      exit_python=False)
      if not overwrite:
        return
      # else generate taxonomy roots

    print('--- Generating taxonomy roots dictionary')

    self._init_map_df()

    roots = {}
    for std_label in self.map_df[self._std_col].unique():
      coarse_labels =  self.get_coarse_labels_for_std_label(std_label)

      # If std_label is a root, find its descending nodes and leafs
      if len(coarse_labels) == 0:
        print(f'\tFound root label: "{std_label}". '
              f'Calculating label tree for "{std_label}" ...')
        roots[std_label] = self.get_dictionary_tree_for_std_label(std_label)

    # Save taxonomy tree
    with open(self._roots_md_file_path, 'w', encoding='utf-8') as json_file:
      json.dump(roots, json_file)

    self.roots = roots


  def get_dictionary_tree_for_std_label(self, std_label, debug=False):
    """Constructs a dictionary tree representing the hierarchical 
    structure of labels for a standard label.

    Parameters
    ----------
    std_label : str
        The standard label for which the dictionary tree is constructed.

    Returns
    -------
    dict
        A dictionary representing the hierarchical structure of labels
        for the given standard label.
    """
    if debug:
      print(std_label)
    # Find fine labels for the standard label, excluding the label itself
    labels = self.get_fine_labels_for_std_label(std_label)
    if len(labels) == 0:
      # If no fine labels found, return a dictionary with the standard
      # label as key and an empty list as value
      return {std_label: labels}
    else:
      dict_tree = {std_label: {}}
      for label in labels:
        # Recursively construct dictionary trees for each fine label and
        # update the main tree
        dict_tree[std_label].update(
          self.get_dictionary_tree_for_std_label(label))
      return dict_tree



  ### - taxonomy utils

  def get_mapped_datasets(self):
    if self.map_df is None:
      self._init_map_df()
    return self.map_df['dataset'].unique().tolist()

  def get_all_dataset_labels(self):
    dataset_labels = {}
    for dataset in self.map_df['dataset'].unique():
      dataset_labels[dataset] = self.map_df.loc[
        self.map_df['dataset'] == dataset]['dataset_label'].unique().tolist()

    return dataset_labels

  def get_default_labels_for_dataset(self, dataset: str):
    """Get all the (default) labels for a mapped dataset.

    Parameters
    ----------
    dataset : str
        The dataset's name as it is mapped in SALT

    Returns
    -------
    list
        List with the default dataset labels.

    Raises
    ------
    ValueError
        Raises ValueError if the dataset is not found in SALT
    """
    if dataset not in self.map_df['dataset'].unique():
      raise ValueError(f'Dataset "{dataset}" not present in the taxonomy')

    return self.map_df.loc[self.map_df['dataset'] == dataset][
      'dataset_label'].unique().tolist()


  def get_mapping_for_std_label(self, std_label):
    """Get the datasets-dataset labels mapping in a dictionary with
    datasets as keys and dataset labels as values for a given standard
    label.

    Parameters
    ----------
    std_label : str
        The standard label to get the mapping for.

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corresponding
        dataset labels as values
    """
    if self.map_df is None:
      self._init_map_df()

    filtered_df = self.map_df.loc[
      self.map_df[self._std_col] == std_label].copy()

    filtered_df.drop_duplicates(subset=['dataset_label', 'dataset'],
                                inplace=True)

    mapping_dict = {}

    for dataset in filtered_df['dataset'].unique():
      mapping_dict[dataset] = filtered_df.loc[
        filtered_df['dataset'] == dataset]['dataset_label'].unique().tolist()

    return mapping_dict


  def get_std_label_from_dataset_label(self, dataset_label):
    """Get standard label given a dataset label

    Parameters
    ----------
    dataset_label : str
        A mapped dataset's label to find its corresponding standard label.

    Returns
    -------
    std
        The standard label corresponding to the input dataset label.

    Raises
    ------
    ValueError
        In case dataset label is not found in the mapping.
    """
    standardized_string = utils.standardize_string(dataset_label)
    if standardized_string in self.map_df[self._std_col].unique():
      return standardized_string

    # Check for filtered datasets
    if not dataset_label in self.map_df['dataset_label'].unique():
      raise ValueError(f'Dataset label {dataset_label} does not exist in '
                       'current mapping...')

    filtered_dbs = self.map_df['dataset'].unique().tolist()
    self.reset_map_df()

    # Get standard labels associated with the given dataset label (coarse/fine)
    std_labels =  self.map_df.loc[self.map_df[
      'dataset_label'] == dataset_label][self._std_col].unique().tolist()

    mapped_label = std_labels[0]
    mapped_label_lvl = len(
      self.map_df.loc[self.map_df[self._std_col] == std_labels[0]][
        'dataset_label'].unique().tolist()
    )
    for std_label in std_labels[1:]:
      current_lvl = len(
        self.map_df.loc[self.map_df[self._std_col] == std_label][
          'dataset_label'].unique().tolist()
      )
      if current_lvl < mapped_label_lvl:
        mapped_label = std_label
        mapped_label_lvl = current_lvl

    # Apply filtering again
    self.filter_by_datasets(filtered_dbs)

    return mapped_label

  def get_mapping_for_dataset_label(self,
                                    dataset_label: str,
                                    return_std_label=False):
    # Find associated standard label
    std_label = self.get_std_label_from_dataset_label(dataset_label)
    # print(std_label)

    # Get the dataset mapping of the standard label
    mapping_dict = self.get_mapping_for_std_label(std_label)

    if return_std_label:
      return mapping_dict, std_label
    else:
      return mapping_dict


  # Function to find all subsets of a specific label
  def get_coarse_labels_for_std_label(self, std_label):
    """Find all the coarser labels (supersets) of the "std_label" label.

    Parameters
    ----------
    std_label : str
        The standard label for which to find its supersets

    Returns
    -------
    list
        list with supersets of "std_label" label
    """
    subsets = []
    label_indices = self.map_df[self._std_col] == std_label
    label_data = self.map_df.loc[label_indices]
    unique_labels = self.map_df[self._std_col].unique()
    for other_label in unique_labels:
      if other_label != std_label:
        other_subset_indices = self.map_df[self._std_col] == other_label
        other_subset_data = self.map_df.loc[other_subset_indices]
        if utils.is_subset(label_data['dataset_label'],
                      other_subset_data['dataset_label']):
          subsets.append(other_label)

    return subsets


  def get_fine_labels_for_std_label(self, std_label):
    """Find all the fine labels (subsets) of the "std_label" label.

    Parameters
    ----------
    std_label : str
        The standard label for which to find its subsets

    Returns
    -------
    list
        list with subsets of "std_label" label
    """
    subsets = []
    label_indices = self.map_df[self._std_col] == std_label
    label_data = self.map_df.loc[label_indices]
    unique_labels = self.map_df[self._std_col].unique()
    for other_label in unique_labels:
      if other_label != std_label:
        other_subset_indices = self.map_df[self._std_col] == other_label
        other_subset_data = self.map_df.loc[other_subset_indices]
        if utils.is_subset(other_subset_data['dataset_label'],
                      label_data['dataset_label']):
          subsets.append(other_label)

    return subsets


  def get_paths_to_label(self, target_label : str):
    """Find path/paths (from coarase to fine) to a standard label.

    Parameters
    ----------
    target_label : str
        The standard label to find paths to.

    Returns
    -------
    list
        list with paths to the target label. Each path is a separate
        list from coarse to fine labels.

    Raises
    ------
    ValueError
        If the given label does not exist in the taxonomy
    """
    if not target_label in self.map_df[self._std_col].unique():
      raise ValueError(f'Label "{target_label}" not present in the taxonomy')

    if target_label in self.roots:
      return [[target_label]]

    return utils.find_all_paths_to_key(dictionary=self.roots,
                                        target_key=target_label)


  def filter_by_datasets(self, dataset_list : list):
    """Filter map_df through a list of one or more datasets.

    Parameters
    ----------
    dataset_list : list
        List with datasets to filter.

    Raises
    ------
    ValueError
        Raise ValueError if a dataset that is not mapped exists in the
        input list.
    """
    if self.map_df is None:
      self._init_map_df()

    for dataset in dataset_list:
      if dataset not in self.map_df['dataset'].unique():
        raise ValueError(f'Dataset {dataset} not found in mapping...')

    # Filter by input datasets
    filtered_df = self.map_df.loc[self.map_df['dataset'].isin(dataset_list)]
    filtered_df = filtered_df.reset_index(drop=True)

    # Replace NaN with empty strings
    filtered_df.replace(pd.NA, '', inplace=True)

    self.map_df = filtered_df


  def reset_map_df(self):
    self._init_map_df()


  def find_datasets_intersection(self, datasets: list):
    """Find the intersection of standard labels for the given datasets.

    Parameters
    ----------
    datasets : list
        List with dataset names.

    Returns
    -------
    list
        List with the intersection of standard labels for the given
        datasets.

    Raises
    ------
    ValueError
        If a given dataset does not exist in the mapping.
    """
    for dataset in datasets:
      if not dataset in self.map_df['dataset'].unique():
        raise ValueError(f'Dataset {dataset} not found in the mapping')
    labels_list = []
    for dataset in datasets:
      dataset_labels = self.map_df.loc[self.map_df['dataset'] == dataset][
        self._std_col].unique().tolist()

      labels_list.append(dataset_labels)

    intersection_set = set(labels_list[0]).intersection(*labels_list)

    return list(intersection_set)


  def get_parent_label_for_std_label(self, std_label : str):
    """Get the parent standard label for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to get its parent.

    Returns
    -------
    list or str
        The parent(s) of the the given standard label

    Raises
    ------
    ValueError
        If the given label does not exist or if it is a root.
    """
    if std_label in self.roots:
      raise ValueError(f'"{std_label}" is a root label.')

    if not std_label in self.map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')

    # Get coarse-to-fine paths
    paths = self.get_paths_to_label(std_label)

    # For each path, get the parent label (second to last)
    parents = []
    for path in paths:
      parents.append(path[-2])

    parents = list(set(parents)) # Ensure no duplicates

    # If there's a single parent return str, else list
    if len(parents) == 1:
      return parents[0]
    else:
      return parents


  def get_children_labels_for_std_label(self, std_label : str):
    """Get list with children label(s) for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to get its children labels

    Returns
    -------
    list
        list with children label(s)

    Raises
    ------
    ValueError
        If the given standard label does not exist in the taxonomy.
    """
    if not std_label in self.map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')
    # Get tree dictionary for the standard label
    dict_tree = utils.find_dict_with_value(self.roots, std_label)

    # Trancate the dictionary to get only the 1st level of nodes
    dict_tree = utils.truncate_dict_at_depth(dict_tree, max_depth=2)

    # If std_label already a leaf, return empty list
    if dict_tree[std_label] == []:
      return []

    # Get children labels
    return list(dict_tree[std_label].keys())


  def get_siblings_labels_for_std_label(self, std_label : str):
    """Get sibling labels for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to look for its siblings.

    Returns
    -------
    dict
        Dict with parent labels as keys and sibling standard labels of
        the given label as values.

    Raises
    ------
    ValueError
        If the label is not present in the taxonomy.
    """
    if not std_label in self.map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')

    # Get parent label/labels
    parent = self.get_parent_label_for_std_label(std_label)

    siblings = {}
    if isinstance(parent, str):
      parent = [parent]

    # For each parent label (if more than 1) get children labels
    # (except std label)
    for p in parent:
      p_siblings = self.get_children_labels_for_std_label(p)
      p_siblings.remove(std_label)
      siblings[p] = p_siblings

    return siblings


  def plot_std_label_mapping(self, std_label, figsize=(8, 8)):
    """Plot mapping tree for standard label. The standard label appears
    as a root node, intermediate nodes represent the mapping datasets
    and leaf nodes represent the dataset labels corresponding to the
    standard label.

    Parameters
    ----------
    std_label : str
        Standardized label
    figsize : tuple, optional
        Figsize, by default (8, 8)
    """
    # --- Create graph edges
    map_dict = {}
    map_dict[std_label] = self.get_mapping_for_std_label(std_label)

    edges = []
    for root, std_label_tree in map_dict.items():
      for dataset, dataset_labels in std_label_tree.items():
        edges.append((f'std: {root}', dataset))
        for dataset_label in dataset_labels:
          edges.append((dataset, f'"{dataset_label}"'))

    # --- Graph parameters
    # Text parameters
    text_linewidth = 1.5
    text_foreground = 'white'
    text_size = 10
    text_rotation = 20

    # Node parameters
    node_size = 1000
    node_color = 'lightblue'
    edge_color = 'lightgrey'

    plt.figure(figsize=figsize)

    # Create the graph and compute the layout
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    pos = graphviz_layout(graph, prog='dot', args='-Grankdir=LR')

    # Determine intermediate nodes
    intermediate_nodes = [node for node in graph.nodes()
                          if graph.in_degree(node) > 0
                          and graph.out_degree(node) > 0]

    # Determine root nodes (nodes with in_degree 0)
    root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    # Create a color map based on the number of intermediate nodes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(intermediate_nodes)))
    intermediate_color_map = dict(zip(intermediate_nodes, colors))

    # Drawing the graph
    nx.draw_networkx(graph, pos,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color)

    # Draw edges from root to intermediate nodes in light grey
    for root in root_nodes:
      root_edges = [(root, target) for target in graph.successors(root)]
      nx.draw_networkx_edges(graph,
                            pos,
                            edgelist=root_edges,
                            edge_color=edge_color,
                            arrows=True)

    # Draw edges with different colors based on the intermediate node
    for node in intermediate_nodes:
      edges = [(node, target) for target in graph.successors(node)]
      edge_color = intermediate_color_map[node]
      nx.draw_networkx_edges(graph,
                            pos,
                            edgelist=edges,
                            edge_color=[edge_color],
                            arrows=True)

    text = nx.draw_networkx_labels(graph, pos)

    # Text config
    for _, t in text.items():
      white_border = withStroke(linewidth=text_linewidth,
                                foreground=text_foreground)
      t.set_path_effects([white_border])
      t.set_size(text_size)
      t.set_rotation(text_rotation)

    # Find the x values of the most left and most right nodes in the graph
    x_values = [coord[0] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    (x,_) = figsize
    offset_horizontal = x*45
    lim_left = min(x_values) - offset_horizontal / 2
    lim_right = max(x_values) + offset_horizontal

    # Find the y values of the topmost and bottommost nodes in the whole graph
    y_values = [coord[1] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    offset_vertical = 70
    lim_top = min(y_values) - offset_vertical
    lim_bottom = max(y_values) + offset_vertical

    plt.xlim(lim_left, lim_right)
    plt.ylim(lim_top, lim_bottom)

    # Create a legend
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=node)
                       for node, color in intermediate_color_map.items()]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
