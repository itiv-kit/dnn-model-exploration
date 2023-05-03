import os
import glob
import argparse
import filetype
import matplotlib.pyplot as plt

from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
from tqdm import tqdm
from multiprocessing import Pool


# https://github.com/scikit-image/scikit-image/issues/4294
def corrupt_image(image_path: str, image_path_base: str, output_directory: str, output_type: str, subset: str, severity_levels: list):
    kind = filetype.guess(image_path)
    if not kind.mime.startswith('image'):
        return False
    img_array = plt.imread(image_path)

    for corruption in get_corruption_names(subset=subset):
        for severity in severity_levels:
            if output_type == 'subdirs':
                output_path = os.path.join(output_directory, corruption, str(severity), os.path.basename(image_path))
            elif output_type == 'filename':
                fname, ext = os.path.splitext(os.path.basename(image_path))
                fn = "{}_{}_{}{}".format(fname, corruption, str(severity), ext)
                output_path_stub = os.path.dirname(os.path.relpath(image_path, image_path_base))
                output_path = os.path.join(output_directory, output_path_stub, fn)
            else:
                raise ValueError("output_type unsupported")

            out_dir = os.path.dirname(output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            corrupted = corrupt(img_array, corruption_name=corruption, severity=severity)

            plt.imsave(output_path, corrupted)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("output_type", choices=['subdirs', 'filename'])
    parser.add_argument("subset", choices=['common', 'validation', 'all', 'noise', 'blue', 'weather', 'digital'])
    parser.add_argument("-s", "--severity", type=int, choices=range(1, 5))
    parser.add_argument("-j", type=int, default=1)
    parser.add_argument("-n", type=int)

    opt = parser.parse_args()
    severity_levels = list(range(1, 6)) if opt.severity is None else [opt.severity]

    # corrupt_image(opt.in_path, opt.out_path, opt.output_type, opt.subset, severity)

    total = opt.n if opt.n is not None else sum([len(files) for r, d, files in os.walk(opt.in_path)])

    p = Pool(opt.j)
    pbar = tqdm(total=total, ascii=True)

    def update_bar(*args):
        pbar.update()

    i = 0

    for filename in glob.glob(os.path.join(opt.in_path, "*"), recursive=True):
        i += 1
        image_path = os.path.join(opt.in_path, filename)
        p.apply_async(corrupt_image, [image_path, opt.in_path, opt.out_path, opt.output_type, opt.subset, severity_levels], callback=update_bar)
        if opt.n and i > opt.n:
            break
    
    p.close()
    p.join()


