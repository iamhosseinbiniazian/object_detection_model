import cv2
import copy
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes
from yolo3.utils import get_random_data
from yolo3.model import preprocess_true_boxes
class BatchGenerator(Sequence):
    def __init__(self, 
        instances, 
        anchors,   
        labels,
        input_shape,
        batch_size=16,
        shuffle=True
                 ,

    ):
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.input_shape         = input_shape
        self.anchors            = anchors

        if shuffle: np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        image_data = []
        box_data = []
        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size
        for train_instance in self.instances[l_bound:r_bound]:
            image, box = get_random_data(train_instance, self.input_shape,self.labels, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes())
        return [image_data, *y_true], np.zeros(self.batch_size)




    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

