import numpy as np
import os
import datetime
from scipy import ndimage
from skimage import io, color

WINDOW_SIZE = 10
IMAGE_SIZE = 167
MIN_OBJECT_AREA = 40
MIN_OBJECT_AMOUNT = 1
ALL_NAMES = ["Nearest Neighbor Filter", "Median Filter", "Midpoint Filter", "Alpha-Trimmed Mean Filter", "Geometric Mean Filter",
             "Harmonic Mean Filter", "Contraharmonic Mean Filter", "Nearest Neighbor Filter + Midpoint Filter", "Modified Median Filter"]

class Method:
    def __init__(self, method_number):
        self.method_number = method_number
        self.name = ""
        self.folder_path = "Unknown Folder"
        self.result_folder_path = "Unknown Folder"
        self.name_list = []

        self.find_method_name(self.method_number)

    def find_method_name(self, method_number):
        self.name = ALL_NAMES[method_number-1]

    def print_info(self):
        i = 1

        print("\n\n=======================")
        print(f"{self.method_number}. {self.name}")
        print(f"Folder path: {self.folder_path}")
        print(f"Result folder path: {self.result_folder_path}")
        print(f"Current images amount: {len(self.name_list)}")
        print(f"All image names:")
        for name in self.name_list:
            print(f"{i}. {name}")
            i += 1
        print("=======================\n\n")

    def check_images(self):
        names = []
        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)

            if image.shape[0] == 167 and image.shape[1] == 167:
                names.append(name)
        
        self.name_list = names

    def set_folder_path(self, folder_path):
        try:
            names = [name for name in os.listdir(folder_path) if name.endswith(".png") and "_stokes_" in name]

            if len(names) == 0:
                print("Your folder is empty!")
            else:
                self.folder_path = folder_path
                self.name_list = names
                self.check_images()
        except:
            print("Your folder doesn't exist")

    def create_result_folder(self):
        current_datetime = datetime.datetime.now()
        result_folder_date = current_datetime.strftime("%d%m%Y")
        result_folder_time = current_datetime.strftime("%H%M%S")
        result_folder_name = f"results_method_{self.method_number}_{result_folder_date}_{result_folder_time}"
        result_folder_path = os.path.join(self.folder_path, result_folder_name)
        os.mkdir(result_folder_path)

        self.result_folder_path = result_folder_path

    def upload_result_image(self, image, name):
        current_datetime = datetime.datetime.now()
        processing_date = current_datetime.strftime("%d%m%Y")
        processing_time = current_datetime.strftime("%H%M%S")
        name = name.replace(".png", "")
        result_image_name = f"{name}_processed_{processing_date}_{processing_time}.png"
        result_image_path = os.path.join(self.result_folder_path, result_image_name)
        io.imsave(result_image_path, image)

    def find_closest_number(self, number_list, target_number):
        closest_number = None
        min_difference = None

        for number in number_list:
            difference = abs(number - target_number)
            if closest_number is None or difference < min_difference:
                closest_number = number
                min_difference = difference

        return closest_number

    def create_window(self, image, r, c, radius):
        min_x = max(0, r - radius // 2)
        max_x = min(image.shape[0], r + radius // 2 + 1)
        min_y = max(0, c - radius // 2)
        max_y = min(image.shape[1], c + radius // 2 + 1)
        
        return image[min_x:max_x, min_y:max_y]

    def find_color(self, image, gray_image, target):
        for r in range(gray_image.shape[0]):
            for c in range(gray_image.shape[1]):
                if gray_image[r][c] == target:
                    return image[r][c]

    def get_mask_indices(self, image):
        channel_list = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
        avg_list = [np.mean(channel) for channel in channel_list]
        max_avg = np.max(avg_list)
        dominant_channel_index = avg_list.index(max_avg)
        second_channel_index = 0
        
        if dominant_channel_index == 0:
            second_channel_index = 2
        
        return dominant_channel_index, avg_list[dominant_channel_index], second_channel_index

    def create_mask(self, image, dominant_channel_index, upper_dominant_limit, second_channel_index):
        dominant_channel = image[:, :, dominant_channel_index]
        second_channel = image[:, :, second_channel_index]
        lower_mean = np.mean(dominant_channel[dominant_channel < upper_dominant_limit])
        lower_dominant_limit = lower_mean + (int(upper_dominant_limit)-int(lower_mean))/2
            
        upper_rfi_mask = dominant_channel > upper_dominant_limit
        lower_rfi_mask = dominant_channel < lower_dominant_limit
        second_rfi_mask = second_channel > 0
        rfi_mask = upper_rfi_mask + lower_rfi_mask + second_rfi_mask
        object_mask, obj_num = self.create_object_mask(image, dominant_channel_index, second_channel_index)
        
        for r in range(object_mask.shape[0]):
            for c in range(object_mask.shape[1]):
                if object_mask[r][c] > 0:
                    rfi_mask[r][c] = False

        return rfi_mask, obj_num
    
    def select_all_objects(self, image, dominant_channel_index, second_channel_index):
        dominant_channel = image[:, :, dominant_channel_index]
        object_mask = dominant_channel == 0
        
        if dominant_channel_index == 1:
            second_channel = image[:, :, second_channel_index]
            second_object_mask = second_channel == 0
            
            for r in range(second_object_mask.shape[0]):
                for c in range(second_object_mask.shape[1]):
                    if second_object_mask[r][c] == True:
                        object_mask[r][c] = False

        return object_mask

    def create_object_mask(self, image, dominant_channel_index, second_channel_index):
        object_mask = self.select_all_objects(image, dominant_channel_index, second_channel_index)
        labeled_mask, num_obj = ndimage.label(object_mask)
        regions_area = ndimage.sum(object_mask, labeled_mask, range(1, num_obj+1))
        
        for region_label, area in enumerate(regions_area, start=1):
            if area < MIN_OBJECT_AREA:
                labeled_mask[labeled_mask == region_label] = 0
                num_obj -= 1
                
        return labeled_mask, num_obj

    def create_blocked_mask(self, image, dominant_channel_index, upper_dominant_limit, second_channel_index):
        dominant_channel = image[:, :, dominant_channel_index]
        second_channel = image[:, :, second_channel_index]
        lower_mean = np.mean(dominant_channel[dominant_channel < upper_dominant_limit])
        lower_dominant_limit = lower_mean + (int(upper_dominant_limit)-int(lower_mean))/2
            
        upper_rfi_mask = dominant_channel > upper_dominant_limit
        lower_rfi_mask = dominant_channel < lower_dominant_limit
        second_rfi_mask = second_channel > 0
        rfi_mask = upper_rfi_mask + lower_rfi_mask + second_rfi_mask
        object_mask = dominant_channel == 0

        return rfi_mask + object_mask

    def check_trim_value(self, window_area, trim_value):
        if trim_value < (window_area/2):
            return trim_value
        else:
            return round((trim_value/2) - 10)

    def try_clear_zone(self, image, gray_image, mask):
        value_list = {}
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r][c] == False:
                    value_list[gray_image[r][c]] = image[r][c]
                
        if len(value_list) == 0:
            return False, value_list

        return True, value_list

    def process_all_images(self):
        if len(self.name_list) > 0:
            match self.method_number:
                case 1:
                    self.process_by_nearest_neighbor_filter()  #Nearest Neighbor Filter
                case 2:
                    self.process_by_median_filter()  #Median Filter
                case 3:
                    self.process_by_midpoint_filter()  #Midpoint Filter
                case 4:
                    self.process_by_alpha_trimmed_mean_filter()  #Alpha-Trimmed Mean Filter
                case 5:
                    self.process_by_geometric_mean_filter()  #Geometric Mean Filter
                case 6:
                    self.process_by_harmonic_mean_filter()  #Harmonic Mean Filter
                case 7:
                    self.process_by_contraharmonic_mean_filter()  #Contraharmonic Mean Filter
                case 8:
                    self.process_by_nearest_neighbor_midpoint_filter()  #Nearest Neighbor Filter + Midpoint Filter
                case 9:
                    self.process_by_modified_median_filter()  #Modified Median Filter

    #Nearest Neighbor Filter
    def process_by_nearest_neighbor_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            blocked_mask = self.create_blocked_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(mask.shape[0]):
                        for c in range(mask.shape[1]):
                            if mask[r][c] == True:
                                original_color = image[r][c]
                                neighbor_color_list = []

                                try:
                                    if r != gray_image.shape[0]:
                                        for i in range(r + 1, gray_image.shape[0]):
                                            if blocked_mask[i][c] == False:
                                                neighbor_color_list.append(image[i][c])
                                                break

                                    if c != gray_image.shape[1]:
                                        for j in range(c + 1, gray_image.shape[1]):
                                            if blocked_mask[r][j] == False:
                                                neighbor_color_list.append(image[r][j])
                                                break
                                    
                                    if r != 0:
                                        for i in range((r - 1) * (-1), 1):
                                            if blocked_mask[abs(i)][c] == False: 
                                                neighbor_color_list.append(image[abs(i)][c])
                                                break

                                    if c != 0:
                                        for j in range((c - 1) * (-1), 1):
                                            if blocked_mask[r][abs(j)] == False:
                                                neighbor_color_list.append(image[r][abs(j)])
                                                break

                                    distances = np.sqrt(np.sum((np.array(neighbor_color_list) - original_color)**2, axis=1))
                                    nearest_color_index = np.argmin(distances)
                                    image[r][c] = neighbor_color_list[nearest_color_index]
                                    mask[r][c] = False
                                    blocked_mask[r][c] = False
                                except:
                                    continue

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Median Filter
    def process_by_median_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                filter_value = np.median(gray_image_neighborhood)
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Midpoint Filter
    def process_by_midpoint_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                filter_value = 0.5 * (np.max(gray_image_neighborhood)+np.min(gray_image_neighborhood))
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Alpha-Trimmed Mean Filter
    def process_by_alpha_trimmed_mean_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                mask_neighborhood = self.create_window(rfi_mask, r, c, WINDOW_SIZE)
                                mn = gray_image_neighborhood.shape[0] * gray_image_neighborhood.shape[1]
                                d = self.check_trim_value(mn, np.sum(mask_neighborhood==True))
                                sorted_values = np.sort(gray_image_neighborhood.flatten())
                                trimmed_values = sorted_values[d:-d]
                                filter_value = np.mean(trimmed_values)
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Geometric Mean Filter
    def process_by_geometric_mean_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                mn = gray_image_neighborhood.shape[0]*gray_image_neighborhood.shape[1]
                                filter_value = np.prod(gray_image_neighborhood)**(1/mn)
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Harmonic Mean Filter
    def process_by_harmonic_mean_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                mn = gray_image_neighborhood.shape[0]*gray_image_neighborhood.shape[1]
                                filter_value = mn / np.sum(1.0 / gray_image_neighborhood)
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Contraharmonic Mean Filter
    def process_by_contraharmonic_mean_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            q_value = -20
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount >= MIN_OBJECT_AMOUNT:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                filter_value = (np.sum(gray_image_neighborhood)**q_value+1)/(np.sum(gray_image_neighborhood)**q_value)
                                closest_number = self.find_closest_number(gray_image_neighborhood.ravel(), filter_value)
                                image[r][c] = self.find_color(image_neighborhood, gray_image_neighborhood, closest_number)
                                gray_image[r][c] = closest_number
                                rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Nearest Neighbor Filter + Midpoint Filter
    def process_by_nearest_neighbor_midpoint_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            blocked_mask = self.create_blocked_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount > 0:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                blocked_mask_neighborhood = self.create_window(blocked_mask, r, c, WINDOW_SIZE)
                                is_clear, clear_dict = self.try_clear_zone(image_neighborhood, gray_image_neighborhood, blocked_mask_neighborhood)

                                if is_clear:
                                    clear_neighborhood = list(clear_dict.keys())
                                    filter_value = 0.5 * (np.max(clear_neighborhood)+np.min(clear_neighborhood))
                                    closest_number = self.find_closest_number(clear_neighborhood, filter_value)
                                    image[r][c] = clear_dict[closest_number]
                                    gray_image[r][c] = closest_number
                                    blocked_mask[r][c] = False
                                    rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")

    #Modified Median Filter
    def process_by_modified_median_filter(self):
        self.create_result_folder()

        for name in self.name_list:
            image = io.imread(self.folder_path + "\\" + name)
            gray_image = color.rgb2gray(image[:, :, :3])
            dominant_channel_index, dominant_limit, second_channel_index = self.get_mask_indices(image)
            rfi_mask, object_amount = self.create_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            blocked_mask = self.create_blocked_mask(image, dominant_channel_index, dominant_limit, second_channel_index)
            is_rfi = False
            privious_rfi_amount = 0

            if object_amount > 0:
                is_rfi = True
            else:
                continue

            while is_rfi:
                rfi_amount = np.sum(rfi_mask==True)

                if rfi_amount == 0 or rfi_amount == privious_rfi_amount:
                    is_rfi = False
                else:
                    for r in range(rfi_mask.shape[0]):
                        for c in range(rfi_mask.shape[1]):
                            if rfi_mask[r][c] == True:
                                gray_image_neighborhood = self.create_window(gray_image, r, c, WINDOW_SIZE)
                                image_neighborhood = self.create_window(image, r, c, WINDOW_SIZE)
                                blocked_mask_neighborhood = self.create_window(blocked_mask, r, c, WINDOW_SIZE)
                                is_clear, clear_dict = self.try_clear_zone(image_neighborhood, gray_image_neighborhood, blocked_mask_neighborhood)

                                if is_clear:
                                    clear_neighborhood = list(clear_dict.keys())
                                    filter_value = np.median(clear_neighborhood)
                                    closest_number = self.find_closest_number(clear_neighborhood, filter_value)
                                    image[r][c] = clear_dict[closest_number]
                                    gray_image[r][c] = closest_number
                                    blocked_mask[r][c] = False
                                    rfi_mask[r][c] = False

                    privious_rfi_amount = rfi_amount

            self.upload_result_image(image, name)
            print(f"{name} is uploaded!")