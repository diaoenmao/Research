import errno
import os
import glob
import shutil

source = 'scratch'
destination = 'result'

def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def main():
	makedir_exist_ok(destination)
	src_file_paths = glob.glob(os.path.join(source, '**/*.pt'), recursive=True)
	for src_file_path in src_file_paths:
	    dst_file_path = os.path.join(destination, os.path.basename(src_file_path))
	    shutil.move(src_file_path, dst_file_path)
	    print('Move {} -> {}'.format(src_file_path, dst_file_path))

if __name__ == "__main__":
    main()