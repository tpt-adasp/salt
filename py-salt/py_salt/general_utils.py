"""General utils
"""
import os
import sys

def find_all_paths_to_key(dictionary : dict,
                          target_key : str,
                          path=None,
                          paths=None):
  """Find all paths to a target key within a nested dictionary.

  Parameters
  ----------
  dictionary : dict
      The nested dictionary to search.
  target_key : str
      The key to search for within the dictionary.
  path : list, optional
      The current path being traversed, by default None.
  paths : list, optional
      A list to store all found paths, by default None.

  Returns
  -------
  list
      A list containing all paths to the target key within the dictionary.
  """
  if path is None:
    path = []
  if paths is None:
    paths = []

  # Iterate through the dictionary items
  for key, value in dictionary.items():
    # Construct the current path
    current_path = path + [key]

    # If the current key matches the target key, add the path to the
    # list of paths
    if key == target_key:
      paths.append(current_path)

    # If the value is a dictionary, recursively search within it
    if isinstance(value, dict):
      find_all_paths_to_key(value, target_key, current_path, paths)

  def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

  def is_list_subset(sublist, superlist):
    return all(item in superlist for item in sublist)

  # Remove duplicates
  paths = [remove_duplicates(path) for path in paths]

  # Remove subset lists (keep only full paths)
  paths = [path for path in paths
           if not any(is_list_subset(path, other) and path != other
                      for other in paths)]

  return paths


def is_subset(set1, set2):
  return set(set1) <= set(set2)


def confirm_overwriting(file_path, exit_python=True):
  """
  Check if a file/directory exists and open prompt to ask for a manual
  confirmation for overwriting.

  Parameters
  ----------
  file_path: str
    file or directory path

  exit_python: bool (default True)
    whether or not exiting python if the answer is to not overwrite
    the existing file/directory

  Returns
  -------
    create_new_file: bool
      in case exit_python=False
  """
  create_new_file = True
  prompt_prefix = '!' * 5 + ' '

  if os.path.isfile(file_path) or os.path.isdir(file_path):

    print(prompt_prefix + file_path + ' already exists')

    key_input = ''
    while key_input.lower() != 'y' and key_input.lower() != 'n':
      key_input = input(prompt_prefix + 'OVERWRITE ANYWAY? [y/n]')

    if key_input.lower() == 'n':
      create_new_file = False

  if exit_python and not create_new_file:
    print('exiting')
    sys.exit()
  else:
    return create_new_file


def set_values_to_empty_lists(d, current_depth, max_depth, keys_to_delete):
  """Recursively set values in the dictionary to empty lists after a
  specified depth, and collect keys to be deleted.

  Parameters
  ----------
  d : dict
      The dictionary to process.
  current_depth : int
      The current depth in the dictionary traversal.
  max_depth : int
      The maximum depth to retain values in the dictionary.
  keys_to_delete : set
      A set that collects keys to be deleted from the dictionary.

  Returns
  -------
  dict or list
      A new dictionary with values set to empty lists beyond the
      specified depth, or the original value if it is not a dictionary.
  """
  if not isinstance(d, dict):
    return d

  if current_depth >= max_depth:
    # Collect keys that should be deleted
    keys_to_delete.update(d.keys())
    return []

  # Recursively call the function for nested dictionaries
  new_dict = {}
  for k, v in d.items():
    if k not in keys_to_delete:
      new_dict[k] = set_values_to_empty_lists(v, current_depth + 1,
                                               max_depth, keys_to_delete)
  return new_dict


def truncate_dict_at_depth(d, max_depth):
  """Truncate the dictionary to a specified depth, setting values
  beyond that depth to empty lists and removing keys that appear
  beyond that depth from the entire dictionary.

  Parameters
  ----------
  d : dict
      The dictionary to be truncated.
  max_depth : int
      The maximum depth to retain values in the dictionary.

  Returns
  -------
  dict
      A new dictionary truncated to the specified depth with values set
      to empty lists beyond that depth. Keys that appear beyond the
      specified depth are removed from the entire dictionary.
  """
  keys_to_delete = set()
  truncated_dict = set_values_to_empty_lists(d, 0, max_depth, keys_to_delete)

  # Remove keys_to_delete from the top-level dictionary
  def remove_keys(d):
    if isinstance(d, dict):
      for key in keys_to_delete:
        if key in d:
          del d[key]
      for _, v in d.items():
        remove_keys(v)
    return d

  truncated_dict = remove_keys(truncated_dict)

  return truncated_dict


def find_dict_with_value(dictionary : dict, target_value):
  """Find a dictionary containing a specific value within a nested dictionary.

  Parameters
  ----------
  dictionary : dict
      The nested dictionary to search.
  target_value : any
      The value to search for within the dictionary.

  Returns
  -------
  dict or None
      A dictionary containing the target value, or None if not found.
  """

  # Check if the target value is a key in the dictionary
  if target_value in dictionary:
    # If the value corresponding to the key is a dictionary, return it
    if isinstance(dictionary[target_value], dict):
      if target_value in dictionary[target_value]:
        return dictionary[target_value]
      else:
        return {target_value: dictionary[target_value]}
    # If the value is an empty list, return it with the key
    elif dictionary[target_value] == []:
      return {target_value: []}

  # Iterate through the dictionary items
  for _, value in dictionary.items():
    # If the value is a dictionary, recursively search within it
    if isinstance(value, dict):
      result = find_dict_with_value(value, target_value)
      # If the value is found within the nested dictionary, return the result
      if result is not None:
        return result
    # If the value matches the target value, return the dictionary containing it
    elif value == target_value:
      return dictionary


def standardize_string(input_string):
  """Standardizes a given input string by removing commas and slashes,
  replacing spaces with underscores, and converting all characters to
  lowercase.

  Parameters
  ----------
  input_string : str
      The input string that needs to be standardized.

  Returns
  -------
  str
      The standardized string with commas and slashes removed, spaces
      replaced by underscores, and all characters converted to lowercase.
  """
  # Remove commas and slashes
  no_commas_slashes = input_string.replace(',', '').replace('/', '')

  # Replace spaces with underscores
  no_spaces = no_commas_slashes.replace(' ', '_')

  # Convert to lowercase
  standardized_string = no_spaces.lower()

  return standardized_string
