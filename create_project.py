import argparse
import fileinput
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--proj', type=str, help='The project name.')
    parser.add_argument('-a', '--app', type=str, help='Name of the first app.')
    args = parser.parse_args()
    project_name = args.proj
    app_name = args.app

    cur_dir = os.getcwd()
    dst_dir = os.path.join(cur_dir, project_name)
    template_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'templates')

    if os.path.exists(dst_dir) and os.path.isdir(dst_dir):
        print(
            "A directory with the same name exists! Use another project name.")
        sys.exit(1)

    shutil.copytree(template_dir, dst_dir)
    shutil.move(
        os.path.join(dst_dir, 'app_dir'), os.path.join(dst_dir, app_name))


if __name__ == '__main__':
    main()
