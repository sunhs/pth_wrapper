import argparse
import fileinput
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str)
    parser.add_argument('-d', type=str)
    args = parser.parse_args()
    project_name = args.n
    project_dir_name = args.d

    cur_dir = os.getcwd()
    dst_dir = os.path.join(cur_dir, project_dir_name)
    template_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'templates')

    shutil.copytree(template_dir, dst_dir)

    # modify main.py
    main_src_path = os.path.join(dst_dir, 'project_name_main.py')
    main_dst_path = os.path.join(dst_dir, 'main.py')
    with open(main_src_path, 'r') as old_f:
        with open(main_dst_path, 'w') as new_f:
            for line in old_f:
                if 'project_name' in line:
                    line = line.replace('project_name', project_name)
                if 'ProjectName' in line:
                    line = line.replace('ProjectName', project_name.title())
                new_f.write(line)
    os.remove(main_src_path)

    # modify dataset
    dataset_src_path = os.path.join(dst_dir, 'project_name_dataset.py')
    dataset_dst_path = os.path.join(dst_dir, project_name + '_dataset.py')
    with open(dataset_src_path, 'r') as old_f:
        with open(dataset_dst_path, 'w') as new_f:
            for line in old_f:
                if 'ProjectName' in line:
                    line = line.replace('ProjectName', project_name.title())
                new_f.write(line)
    os.remove(dataset_src_path)

    # modify trainer
    trainer_src_path = os.path.join(dst_dir, 'project_name_trainer.py')
    trainer_dst_path = os.path.join(dst_dir, project_name + '_trainer.py')
    with open(trainer_src_path, 'r') as old_f:
        with open(trainer_dst_path, 'w') as new_f:
            for line in old_f:
                if 'ProjectName' in line:
                    line = line.replace('ProjectName', project_name.title())
                new_f.write(line)
    os.remove(trainer_src_path)


if __name__ == '__main__':
    main()
