import sys
sys.path.append('..')
import argparse
from ccpg.sast.src_parser import c_parser
from ccpg.util.helper import visualize_helper as c_visualize_helper
from javacpg.sast.src_parser import java_parser
from javacpg.util.helper import visualize_helper as java_visualize_helper

def visualize_c(path, fig_name):
    func_list = c_parser(path)
    if len(func_list) != 1:
        print('Error: isfactor.c should have only one function.')
        exit(-1)
    for func in func_list:
        c_visualize_helper(func, fig_name)

def visualize_java(path, fig_name):
    func_list = java_parser(path)
    if len(func_list) != 1:
        print('Error: isfactor.java should have only one function.')
        exit(-1)
    for func in func_list:
        java_visualize_helper(func, fig_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='c', help='c or java')
    parser.add_argument('--path', type=str, default='./isfactor.c', help='path to the source code')
    parser.add_argument('--fig_name', type=str, default='./example_c', help='path to the figure')

    args = parser.parse_args()
    
    if args.lang == 'c':
        visualize_c(args.path, args.fig_name)
    elif args.lang == 'java':
        visualize_java(args.path, args.fig_name)
    else:
        print('Error: language should be c or java.')