import logging

import numpy as np
import scipy
import torch.utils.data
import pathlib
import re
import numpy
import os
import numpy.linalg
import numpy
from enum import Enum
import torch.utils.data
import pathlib
import librosa
import torchaudio
import torchaudio.functional
import tqdm


class OAEDataset(torch.utils.data.Dataset):
    def __init__(self, dset_path, noise_path):
        self.dset_path = dset_path
        self.noise_path = noise_path

        self.data = self.__load_dset()


    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, item):
        # datapoints_with_class = self.data[item]
        class_number = self.data[1][item]
        datapoints = self.data[0][item]
        return torch.from_numpy(datapoints), torch.tensor(class_number)

    def __load_dset(self):
        if not self.dset_path:
            logging.error("Dset path is empty!")
            raise ValueError("Dset path is empty!")

        # create classes
        classes = [pathlib.Path(x[0]).stem for x in os.walk(self.dset_path)][1:]
        dset = []
        dset_pathlib_obj = pathlib.Path(self.dset_path)
        dset_files = list(dset_pathlib_obj.rglob("*txt"))
        dset_files.extend(list(dset_pathlib_obj.rglob("*TXT")))
        class_assigments = []
        # that will be slow!
        for file in dset_files:
            logging.info(f"Reading file {file}.")
            for idx, class_name in enumerate(classes):
                if class_name in str(file):
                    txt_file_text = file.read_text()
                    split_to_entries = txt_file_text.split("\n")[3:-1]
                    data_points = []
                    whitespace_re = re.compile(r"\s+")
                    for entry in split_to_entries:
                        data_point = whitespace_re.sub(" ", entry).strip().split(" ")[1]
                        data_points.append(float(data_point))
                    # last element will be the class
                    # normalized_datapoints = self.normalize(data_points, 0, 1)
                    # normalized_datapoints.append(idx)
                    dset.append(data_points)
                    class_assigments.append(idx)
        dset = numpy.array(dset, dtype=numpy.float32)
        dset = (dset-numpy.min(dset))/(numpy.max(dset)-numpy.min(dset))
        # class_assigments = (class_assigments-numpy.min(class_assigments))/(numpy.max(class_assigments)-numpy.min(class_assigments))

        return dset, class_assigments


class DatasetType(Enum):
    DP = 1,
    TE = 2


class OAEMatDataset(torch.utils.data.Dataset):
    def __init__(self, dset_path, type: DatasetType):
        self.dset_path = dset_path
        self.dset_type = type
        self.data, self.class_assigments = self.__load_dset()

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float).squeeze(0), torch.tensor(self.class_assigments[index])

    def __len__(self):
        return len(self.data)

    def __load_dset(self):
        dset_path_obj = pathlib.Path(self.dset_path)

        if dset_path_obj.is_file():
            raise ValueError("Target dataset path should be a directory, not file.")

        if self.dset_type == DatasetType.DP:
            glob_pattern = "*DP*mat"
        elif self.dset_type == DatasetType.TE:
            glob_pattern = "*TE*mat"
        else:
            raise ValueError(f"Invalid dataset {self.dset_type} type. Either DP or TE are correct.")

        mat_files = dset_path_obj.glob(glob_pattern)
        class_assigments = []
        data = []
        for i, mat_file in enumerate(mat_files):
            split_filename = str(mat_file.stem).split(" ")
            try:
                loaded_mat = scipy.io.loadmat(str(mat_file))["output_teoae"][0]
                converted = [float(item) for item in loaded_mat]  # mat files are weird af
                converted = converted / numpy.max(numpy.abs(converted))
            except KeyError as e:
                print("Invalid matfile, it should contain output_teoae key.")
                raise e
            # for cross entropy loss, classes need to span from 0 to N-1
            class_number = int(split_filename[0][1:]) - 1
            class_assigments.append(class_number)
            data.append(converted)

        print(max(set(class_assigments)))
        return data, class_assigments



class OAESTFTMatDataset(torch.utils.data.Dataset):
    def __init__(self, dset_path, type: DatasetType):
        self.dset_path = dset_path
        self.dset_type = type
        # self.noise_path = noise_path
        # self.noises = self.__load_noises()
        self.data, self.class_assigments = self.__load_dset()

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float).unsqueeze(0), torch.tensor(self.class_assigments[index])

    def __len__(self):
        return len(self.data)

    def __load_dset(self):
        dset_path_obj = pathlib.Path(self.dset_path)

        if dset_path_obj.is_file():
            raise ValueError("Target dataset path should be a directory, not file.")

        if self.dset_type == DatasetType.DP:
            glob_pattern = "*DP*mat"
        elif self.dset_type == DatasetType.TE:
            glob_pattern = "*TE*mat"
        else:
            raise ValueError(f"Invalid dataset {self.dset_type} type. Either DP or TE are correct.")

        mat_files = dset_path_obj.glob(glob_pattern)
        class_assigments = []
        data = []
        for i, mat_file in enumerate(mat_files):
            split_filename = str(mat_file.stem).split(" ")
            try:
                loaded_mat_a = scipy.io.loadmat(str(mat_file))["Data"]["A"][0][0]  # IDK MAN
                loaded_mat_b = scipy.io.loadmat(str(mat_file))["Data"]["B"][0][0]
                concated = np.concatenate([loaded_mat_a, loaded_mat_b])
                converted = np.float64(concated).T
                converted = converted / numpy.max(numpy.abs(converted))
                # print(converted.shape)
                for item in converted:
                    item = item[182:]
                    # for noise in self.noises:
                    # stft_val = librosa.stft(item)
                    # augmented_item = torchaudio.functional.add_noise(item, noise, torch.tensor(3))
                    # for cross entropy loss, classes need to span from 0 to N-1
                    data.append(item)
                    class_number = int(split_filename[0][0:]) - 1
                    class_assigments.append(class_number)
                        # print(numpy.abs(stft_val).flatten().shape)
            except KeyError as e:
                print("Invalid matfile, it should contain output_teoae key.")
                raise e


        print(max(set(class_assigments)))
        return data, class_assigments


class OAEAugmentedMatDataset(torch.utils.data.Dataset):
    def __init__(self, dset_path, type: DatasetType, noise_path=""):
        self.max_class_number = 0
        self.dset_path = dset_path
        self.dset_type = type
        self.noise_path = noise_path
        if self.noise_path is not "":
            self.noises = self.__load_noises()
        self.mat_files = []
        self.data, self.class_assigments, self.filename = self.__load_dset()

    def __remove_leading_zeros(self, noise_file):
        new_noise = []
        for item in noise_file:
            if item == 0:
                continue
            new_noise.append(item.item())
        return new_noise

    def __load_noises(self):
        noise_exts = ["wav", "flac"]
        noise_files = []
        # noise_files_objs = []
        path_obj = pathlib.Path(self.noise_path)
        for noise_ext in noise_exts:
            found_noise_files = list(path_obj.glob(f"*{noise_ext}"))
            for found_noise_file in found_noise_files:
                loaded_obj = torchaudio.load(str(found_noise_file))
                #normalized_noise = torch.nn.functional.normalize(torch.tensor((loaded_obj[0][0].numpy())), dim = 0)
                #normalized_noise = torch.tensor((loaded_obj[0][0].numpy() - numpy.mean(loaded_obj[0][0].numpy())) / numpy.std(numpy.abs(loaded_obj[0][0].numpy())), dtype=torch.float)
                resampled_obj = torchaudio.functional.resample(torch.tensor((loaded_obj[0][0].numpy())), 44100, 48000)
                # resampled_obj = self.__remove_leading_zeros(resampled_obj)
                noise_files.append(resampled_obj)
        #noise_files[2] = noise_files[2][2240:]
        return noise_files

    def __getitem__(self, index):
        encoding_vector = np.zeros(self.max_class_number)
        class_assigment = self.class_assigments[index]
        encoding_vector[class_assigment] = 1
        return self.data[index].unsqueeze(0), torch.tensor(encoding_vector, dtype=torch.float16)

    def __len__(self):
        return len(self.data)



    def __load_dset(self):
        dset_path_obj = pathlib.Path(self.dset_path)

        if dset_path_obj.is_file():
            raise ValueError("Target dataset path should be a directory, not file.")

        if self.dset_type == DatasetType.DP:
            glob_pattern = "*DP*mat"
        elif self.dset_type == DatasetType.TE:
            glob_pattern = "*TE*mat"
        else:
            raise ValueError(f"Invalid dataset {self.dset_type} type. Either DP or TE are correct.")

        mat_files = dset_path_obj.glob(glob_pattern)
        self.mat_files = list(mat_files)
        class_assigments = []
        data = []
        filenames = []
        for i, mat_file in enumerate(self.mat_files):
            split_filename = str(mat_file.stem).split(" ")
            try:
                if "A" in str(mat_file):
                    loaded = scipy.io.loadmat(str(mat_file))["cut_teoae_A"]  # IDK MAN
                else:
                    loaded = scipy.io.loadmat(str(mat_file))["cut_teoae_B"]
                converted = np.float64(loaded.T)
                converted = (converted-numpy.min(converted))/(numpy.max(converted)-numpy.min(converted))

                #c
                #converted = converted.T
                # print(converted.shape)

                for item in converted:
                    item = torch.tensor(item, dtype=torch.float)
                    item = torch.nan_to_num(item, nan=6.10352e-05, posinf=1, neginf=0)
                        # augment data with noise
                    if self.noise_path is not "":
                        for noise in self.noises:
                            # stft_val = librosa.stft(item)
                            for db_val in [4, 20]:
                                augmented_item = torchaudio.functional.add_noise(item, noise[0:len(item)], torch.tensor(db_val))
                                augmented_item = augmented_item.numpy()
                                augmented_item = np.float64((augmented_item - numpy.min(augmented_item)) / (numpy.max(augmented_item) - numpy.min(augmented_item)))
                                augmented_item = torch.tensor(augmented_item, dtype=torch.float16)
                                augmented_item = torch.nan_to_num(augmented_item, nan=6.10352e-05, posinf=1, neginf=0)
                                data.append(augmented_item)
                                # for cross entropy loss, classes need to span from 0 to N-1
                                class_number = int(split_filename[0][0:]) - 13
                                class_assigments.append(class_number)
                                # print(numpy.abs(stft_val).flatten().shape)
                                filenames.append(mat_file.stem)

                    # add noise-less item
                    item = item.to(torch.float16)
                    item = torch.nan_to_num(item, nan=6.10352e-05, posinf=1, neginf=0)
                    data.append(item)
                    class_number = int(split_filename[0]) - 13
                    class_assigments.append(class_number)
                    filenames.append(mat_file.stem)

            except KeyError as e:
                print("Invalid matfile, it should contain Data:A or Data:B key.")
                raise e


        print(max(set(class_assigments)) +1 )
        self.max_class_number = max(set(class_assigments)) + 1
        return data, class_assigments, filenames


class OAEAugmentedMatDatasetTest(torch.utils.data.Dataset):
    def __init__(self, dset_path, type: DatasetType):
        self.max_class_number = 0
        self.dset_path = dset_path
        self.dset_type = type
        self.mat_files = []
        self.data, self.class_assigments, self.filename = self.__load_dset()

    def __getitem__(self, index):
        encoding_vector = np.zeros(self.max_class_number)
        class_assigment = self.class_assigments[index]
        encoding_vector[class_assigment] = 0
        return self.data[index].unsqueeze(0), torch.tensor(encoding_vector, dtype=torch.float16)

    def __len__(self):
        return len(self.data)



    def __load_dset(self):
        dset_path_obj = pathlib.Path(self.dset_path)

        if dset_path_obj.is_file():
            raise ValueError("Target dataset path should be a directory, not file.")

        if self.dset_type == DatasetType.DP:
            glob_pattern = "*DP*mat"
        elif self.dset_type == DatasetType.TE:
            glob_pattern = "*TE*mat"
        else:
            raise ValueError(f"Invalid dataset {self.dset_type} type. Either DP or TE are correct.")

        mat_files = dset_path_obj.glob(glob_pattern)
        self.mat_files = list(mat_files)
        class_assigments = []
        data = []
        filenames = []
        for i, mat_file in enumerate(self.mat_files):
            split_filename = str(mat_file.stem).split(" ")
            try:
                if "A" in str(mat_file):
                    loaded = scipy.io.loadmat(str(mat_file))["cut_teoae_A"]  # IDK MAN
                else:
                    loaded = scipy.io.loadmat(str(mat_file))["cut_teoae_B"]
                converted = np.float64(loaded.T)
                converted = (converted-numpy.min(converted))/(numpy.max(converted)-numpy.min(converted))

                #c
                #converted = converted.T
                # print(converted.shape)

                for item in converted:
                    item = torch.tensor(item, dtype=torch.float)
                    item = torch.nan_to_num(item, nan=6.10352e-05, posinf=1, neginf=0)
                        # augment data with noise

                    # add noise-less item
                    item = item.to(torch.float16)
                    item = torch.nan_to_num(item, nan=6.10352e-05, posinf=1, neginf=0)
                    data.append(item)
                    class_number = 0
                    class_assigments.append(class_number)
                    filenames.append(mat_file.stem)

            except KeyError as e:
                print("Invalid matfile, it should contain Data:A or Data:B key.")
                raise e


        print(max(set(class_assigments)) + 1)
        self.max_class_number = 24
        return data, class_assigments, filenames