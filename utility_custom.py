'''

'''
import pathlib as pt

def output_control(path_dir):
    path_output = pt.Path(path_dir)
    if path_output.exists() is True:
        if path_output.is_dir():
            return 0
        else: 
            print('The path is not a directory: Error\n')
            return 1
    else:
        path_output.mkdir()
        return 2
