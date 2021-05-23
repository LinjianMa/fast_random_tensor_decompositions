import numpy as np
import os
from os.path import dirname, join
from scipy.io import loadmat
from .utils import download_unzip_data, load_images_from_folder


def coil_100(tenpy):
    """
    Columbia University Image Library (COIL-100)
    References:
        http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
        Columbia Object Image Library (COIL-100), S. A. Nene, S. K. Nayar and H. Murase, 
        Technical Report CUCS-006-96, February 1996.

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors')

    def create_bin():
        urls = [
            'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
        ]
        zip_names = ['coil-100.zip']
        file_name = 'coil-100/'
        download_unzip_data(urls, zip_names, data_dir)

        coil_folder = join(data_dir, file_name)
        nonimage_names = ['convertGroupppm2png.pl', 'convertGroupppm2png.pl~']
        for file in nonimage_names:
            nonimage_path = join(coil_folder, file)
            if os.path.isfile(nonimage_path):
                os.remove(nonimage_path)

        pixel = load_images_from_folder(coil_folder)
        pixel_out = np.reshape(pixel, (7200, 128, 128, 3)).astype(float)

        output_file = open(join(data_dir, 'coil-100.bin'), 'wb')
        print("Print out pixels ......")
        pixel_out.tofile(output_file)
        output_file.close()

    if not os.path.isfile(join(data_dir, 'coil-100.bin')):
        create_bin()
    pixels = np.fromfile(join(data_dir, 'coil-100.bin'), dtype=float).reshape(
        (7200, 128, 128, 3))
    return pixels[:, :, :, 0]#pixels[:, :, :, :]


def time_lapse_images(tenpy):
    """
    Time-Lapse Hyperspectral Radiance Images of Natural Scenes 2015
    Datasets are under the CCBY license (http://creativecommons.org/licenses/by/4.0/).
    References:
        https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/Time-Lapse_HSIs_2015.html
        Foster, D.H., Amano, K., & Nascimento, S.M.C. (2016). Time-lapse ratios of cone excitations 
        in natural scenes. Vision Research, 120, 45-60.doi.org/10.1016/j.visres.2015.03.012

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors')

    def create_bin():
        urls = [
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1140.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1240.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1345.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1441.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1600.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1637.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1745.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1845.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1941.zip'
        ]
        zip_names = [
            'nogueiro_1140.zip', 'nogueiro_1240.zip', 'nogueiro_1345.zip',
            'nogueiro_1441.zip', 'nogueiro_1600.zip', 'nogueiro_1637.zip',
            'nogueiro_1745.zip', 'nogueiro_1845.zip', 'nogueiro_1941.zip'
        ]
        download_unzip_data(urls, zip_names, data_dir)

        x = []
        print("Loading 1st dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1140.mat')['hsi'])
        print("Loading 2nd dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1240.mat')['hsi'])
        print("Loading 3rd dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1345.mat')['hsi'])
        print("Loading 4th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1441.mat')['hsi'])
        print("Loading 5th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1600.mat')['hsi'])
        print("Loading 6th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1637.mat')['hsi'])
        print("Loading 7th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1745.mat')['hsi'])
        print("Loading 8th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1845.mat')['hsi'])
        print("Loading 9th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1941.mat')['hsi'])
        x = np.asarray(x).astype(float)
        print(x.shape)

        output_file = open(join(data_dir, 'time-lapse.bin'), 'wb')
        print("Print out data ......")
        x.tofile(output_file)
        output_file.close()

    if not os.path.isfile(join(data_dir, 'time-lapse.bin')):
        create_bin()
    pixels = np.fromfile(join(data_dir, 'time-lapse.bin'),
                         dtype=float).reshape((9, 1024, 1344, 33))
    return pixels[0, :, :, :]
