import numpy as np
import random
from active_sampling import Similarity_measures
class preprocess():
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.sorted_by_class = {}
        self.order_by_class()
        self.preselected_idx ={}

    def order_by_class(self):
        for class_id in range(0,100):
            indexes = [index for index in range(len(self.labels)) if self.labels[index] == class_id]
            self.sorted_by_class[f'{class_id}'] = indexes

    def selection_process(self,number_of_images):
        """For each class select n_number of images that are the furthest away from preselected img"""
        for class_id in range(0,100):
            all_indices = self.sorted_by_class[f'{class_id}']
            picture_to_compare = all_indices[0]
            distance_list = []
            for idx,comparing_img in enumerate(all_indices):
                if idx == 0:
                    distance_list.append([0,comparing_img])
                    continue
                distance_list.append([Similarity_measures(self.images[picture_to_compare],self.images[comparing_img]).peek_signal_to_noise_ratio(),comparing_img])
            distance_list.sort(reverse=True)
            self.preselected_idx[f'{class_id}'] = [distance_list[i][1] for i in range(0,number_of_images)]
            #selected_images = [self.images[distance_list[i][1]] for i in range(0,number_of_images)]
            #selected_labels = [self.labels[distance_list[i][1]] for i in range(0,number_of_images)]

    def get_corresponding_images_and_labels(self):

        for class_id in range(0,100):
            combined_list = [x for y in list(self.preselected_idx.values()) for x in y]
            random.shuffle(combined_list)
            imgs = [self.images[idx] for idx in combined_list]
            lbls = [self.labels[idx] for idx in combined_list]
        return imgs,lbls














