'''

'''
import pathlib as pt

def output_control(path_dir):
    path_output = pt.Path(path_dir)

    for parent_path in path_output.parents:
        if parent_path.exists() is True:
            if parent_path.is_dir() is False:
                print('The partent path is not a directory: Error\n')
                return 0
        else:
            parent_path.mkdir()
            
    if path_output.exists() is True:
        if path_output.is_dir() is False:
            print('The path is not a directory: Error\n')
            return 0
    else:
        path_output.mkdir()
        if (path_output.exists() is True) and (path_output.is_dir() is False):
            print('The path is not a directory: Error\n')
            return 0
            
