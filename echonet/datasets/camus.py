from torch.utils.data import Dataset
import pathlib
import os
import collections
import SimpleITK as sitk
import numpy as np
import cv2

'''
This function reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions 
    # to get axis in the order z,y,x, 
    ct_scan = sitk.GetArrayFromImage(itkimage)
    # also add channel dimension with reshape
    ct_scan = ct_scan.reshape(1, ct_scan.shape[0], ct_scan.shape[1], 
                              ct_scan.shape[2])
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

class Camus(Dataset):
    """Camus Dataset.
     Args:
     root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
     split (string): One of {"train", "val", "test", "external_test"}
     target_type (string or list, optional): Type of target to use,
         ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
         ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
         or ``SmallTrace''
         Can also be a list to output a tuple with all specified target types.
         The targets represent:
             ``Filename'' (string): filename of video
             ``EF'' (float): ejection fraction
             ``EDV'' (float): end-diastolic volume
             ``ESV'' (float): end-systolic volume
             ``LargeIndex'' (int): index of large (diastolic) frame in video
             ``SmallIndex'' (int): index of small (systolic) frame in video
             ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
             ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
             ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                 value of 0 indicates pixel is outside left ventricle
                          1 indicates pixel is inside left ventricle
             ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                 value of 0 indicates pixel is outside left ventricle
                          1 indicates pixel is inside left ventricle
         Defaults to ``EF''.
     mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
         Used for normalizing the video. Defaults to 0 (video is not shifted).
     std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
         Used for normalizing the video. Defaults to 0 (video is not scaled).
     length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
         Defaults to 16.
     period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
         Defaults to 2.
     max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
         long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
         Defaults to 250.
     clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
         Defaults to 1.
     pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
         and a window of the original size is taken. If ``None'', no padding occurs.
         Defaults to ``None''.
     noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
         Defaults to ``None''.
     target_transform (callable, optional): A function/transform that takes in the target and transforms it.
     external_test_location (string): Path to videos to use for external testing.
 """
    def __init__(self, root="../data/Camus/",
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 resize=(112,112), 
                 include_poor_quality=False):
        self.resize = resize
        self.mean = mean
        self.std =std
        self.root = root
        self.split = split
        self.noise = noise
        self.length = length
        self.max_length = max_length
        self.period = period
        self.pad = pad
        self.target_transform = target_transform
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.include_poor_quality = include_poor_quality
        if self.split == "train":
            #training set directory
            self.split_dir = pathlib.Path(os.path.join(root, "training"))
            # patient folder names = patient0xxx
            self.patients = sorted(os.listdir(self.split_dir))
        elif self.split == "test":
            self.split_dir = pathlib.Path(os.path.join(root, "testing"))
            self.patients = sorted(os.listdir(self.split_dir))
        elif self.split == "val":
            self.split_dir = pathlib.Path(os.path.join(root, "validation"))
            self.patients = sorted(os.listdir(self.split_dir))
        # filter out empty dirs and poor quality patients if asked
        if self.include_poor_quality:
            self.patients = [self.patients[i] for i in range(len(self.patients)) 
                             if (len(os.listdir(os.path.join(self.split_dir, 
                                                        self.patients[i])))) > 0]
        else:
            self.patients = [self.patients[i] for i in range(len(self.patients)) 
                             if (len(os.listdir(os.path.join(self.split_dir, 
                                                        self.patients[i]))) > 0
                             and not self.has_poor_quality(self.patients[i]) 
                             == False)]           
        
    def __getitem__(self, idx):
        # initialise subdict for each patient
        d = {}
        # list for 2 videos (2CH, 4CH)
        videos = []
        # subdictionary of info2
        temp = {}
        if self.split != "val":
            patient_folder_name = "patient" + str(idx+1).zfill(4)
        else:
            patient_folder_name = "patient" + str(idx+1+400).zfill(4)
        with open(os.path.join(self.split_dir, patient_folder_name,
                               "Info_2CH.cfg")) as f:
            for line in f:
                key, value = line.strip().split(': ')
                temp[key] = value
        #append to main dict
        d["info2"] = temp
    
        #subdictionary of info4
        temp = {}
        with open(os.path.join(self.split_dir, patient_folder_name,
                               "Info_4CH.cfg")) as f:
            for line in f:
                key, value = line.strip().split(': ')
                temp[key] = value
        #append to main dict
        d["info4"] = temp
        
        # read images and videos
        prefixes = ["_2CH_ED.mhd", "_2CH_ED_gt.mhd", "_2CH_ES.mhd", 
                    "_2CH_ES_gt.mhd", "_2CH_sequence.mhd", "_4CH_ED.mhd", 
                    "_4CH_ED_gt.mhd", "_4CH_ES.mhd", "_4CH_ES_gt.mhd", 
                    "_4CH_sequence.mhd"]
        dict_names = ["ed2", "ed2_gt", "es2", "es2_gt", "sequence2",
                  "ed4", "ed4_gt", "es4", "es4_gt", "sequence4"]
        for (prefix, name) in zip(prefixes, dict_names):
            #subdictionary of mhd
            temp = {}
            img, origin, spacing = load_itk(os.path.join(self.split_dir,
                                   patient_folder_name, patient_folder_name
                                   + prefix))
            channels, frames, height, width  = img.shape
            # downsampling and normalisation depending on type of image
            if not "gt" in prefix:
                img = img.astype("float32")
                res = np.empty((channels, frames, self.resize[0], 
                                       self.resize[1]), dtype=np.float32)
                # downsampling to 112x112 with opencv (cubic interpolation)
                for f in range(frames):
                    for c in range(channels):
                        res[c, f, :, :] = cv2.resize(img[c, f, :, :], 
                                         dsize=self.resize, 
                                         interpolation=cv2.INTER_CUBIC)
                # normalisation 
                res = np.repeat(res, 3, axis=0)
                channels = res.shape[0]
                if isinstance(  self.mean, (float, int)):
                    res -= self.mean
                else:
                    res -= self.mean.reshape(channels, 1, 1, 1)
                if isinstance(self.std, (float, int)):
                    res /= self.std
                else:
                    res /= self.std.reshape(channels, 1, 1, 1)
            # downsampling without normalisation for gt
            else:
                res = np.empty((channels, frames, self.resize[0], 
                                self.resize[1]), dtype=np.uint8)
                # only nearest neighbour downsampling for GT
                res[0, f, :, :] = cv2.resize(img[0, f, :, :], 
                                        self.resize, 
                                        interpolation=cv2.INTER_NEAREST)
            img = res
            #print(img.shape)
            # extra video processing - Echonet c/p
            video_flag = False
            if "sequence" in prefix:
                video_flag = True
                video = img
                # Add simulated noise (black out random pixels)
                # 0 represents black at this point (video has not been normalized yet)
                if self.noise is not None:
                    n = video.shape[1] * video.shape[2] * video.shape[3]
                    ind = np.random.choice(n, round(self.noise * n), replace=False)
                    f = ind % video.shape[1]
                    ind //= video.shape[1]
                    i = ind % video.shape[2]
                    ind //= video.shape[2]
                    j = ind
                    video[:, f, i, j] = 0
                # Set number of frames
                c, f, h, w = video.shape
                if self.length is None:
                    # Take as many frames as possible
                    length = f // self.period
                else:
                    # Take specified number of frames
                    length = self.length
        
                if self.max_length is not None:
                    # Shorten videos to max_length
                    length = min(length, self.max_length)
        
                if f < length * self.period:
                    # Pad video with frames filled with zeros if too short
                    # 0 represents the mean color (dark grey), since this is after normalization
                    video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
                    c, f, h, w = video.shape  # pylint: disable=E0633
                if self.pad is not None:
                    # Add padding of zeros (mean color of videos)
                    # Crop of original size is taken out
                    # (Used as augmentation)
                    c, l, h, w = video.shape
                    tem = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad),
                                    dtype=video.dtype)
                    # pylint: disable=E1130
                    tem[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  
                    i, j = np.random.randint(0, 2 * self.pad, 2)
                    video = tem[:, :, i:(i + h), j:(j + w)]
                videos.append(video)
            if video_flag:
                temp["video"] = video
            else:
                temp["img"] = res
            temp["origin"] = origin
            temp["spacing"] = spacing
            d[name] = temp
        # Gather targets
        target = []
        for t in self.target_type:
            if t == "EF":
                target.append(float(d["info4"]["LVef"]))
                target = np.array(target, dtype = "float32")
            elif t == "EDV":
                target.append(float(d["info4"]["LVedv"]))
            elif t == "ESV":
                target.append(float(d["info4"]["LVesv"]))
            elif t == "Filename":
                target.append(patient_folder_name)
            elif t == "LargeIndex2":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(d["info2"]["ED"]))
            elif t == "SmallIndex2":
                # Largest (diastolic) frame is first
                target.append(int(d["info2"]["ES"]))
            elif t == "LargeFrame2":
                target.append(d["ed2"]["img"])
            elif t == "SmallFrame2":
                target.append(d["es2"]["img"])
            elif t == "LargeTrace2":
                target.append(d["ed2_gt"]["img"])
            elif t == "SmallTrace2":
                target.append(d["es2_gt"]["img"])
            elif t == "LargeIndex4":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(d["info4"]["ED"])
            elif t == "SmallIndex4":
                # Largest (diastolic) frame is first
                target.append(d["info4"]["ES"])
            elif t == "LargeFrame4":
                target.append(d["ed4"]["img"])
            elif t == "SmallFrame4":
                target.append(d["es4"]["img"])
            elif t == "LargeTrace4":
                target.append(d["ed4_gt"]["img"])
            elif t == "SmallTrace4":
                target.append(d["es4_gt"]["img"])
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        return videos[0], target
      
    def __len__(self):
        return len(self.patients)
    
    def has_poor_quality(self, patient):
        temp={}
        with open(os.path.join(self.split_dir, patient,
                                   "Info_4CH.cfg")) as f:
            for line in f:
                key, value = line.strip().split(': ')
                temp[key] = value
        return (temp["ImageQuality"] == "Poor")