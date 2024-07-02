import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import albumentations as A
from tqdm import tqdm



def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 600,
    slice_width: int = 600,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.1,
) -> list[list[int]]:


    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def get_rectangle_params_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_bboxes(
    plot_ax,
    bboxes,
    class_labels,
    get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox,
):
    for bbox, label in zip(bboxes, class_labels):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left, width, height, linewidth=4, edgecolor="green", fill=False,
        )

        rx, ry = rect_1.get_xy()

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.annotate(label, (rx+width, ry+height), color='white', fontsize=20)

def show_image(image, bboxes=None, class_labels=None, draw_bboxes_fn=draw_bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if bboxes:
        draw_bboxes_fn(ax, bboxes, class_labels)

    plt.show()

def convert_yolo_bbox(
    coord : list,
    img_size: tuple,
) -> tuple:


    dh, dw = img_size
    cls, x, y, w, h = map(float, coord.split(' '))


    x1 = int((x - w / 2) * dw)
    x2 = int((x + w / 2) * dw)
    y1 = int((y - h / 2) * dh)
    y2 = int((y + h / 2) * dh)

    if x1 < 0:
       x1 = 0
    if x2 > dw - 1:
        x2 = dw - 1
    if y1 < 0:
        y1 = 0
    if y2 > dh - 1:
        y2 = dh - 1

    return (cls, [x1,y1,x2,y2])

def convert_voc_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def load_labels(
    labels: str,
    image_size: tuple,
)-> tuple:
    bboxs , clss = [], []

    fl = open(labels, 'r')
    data = fl.readlines()
    fl.close()

    for dt in data:

        cls, box_voc = convert_yolo_bbox(dt,image_size)

        bboxs.append(box_voc)
        clss.append(cls)

    return (np.array(clss), np.array(bboxs), data)

def past_exists(path):
    
     if not os.path.isdir(path): # vemos de este diretorio ja existe
        os.mkdir(path) # aqui criamos a pasta caso nao exista



def crop_dataset(path_img : str,
                 path_labels : str,
                 size_crop : tuple,
                 save_dir_img: str,
                save_dir_labels: str,
                save_dir_no_img: str,
                save_dir_no_labels: str
) -> int:
    
    past_exists(save_dir_img)
    past_exists(save_dir_labels)
    past_exists(save_dir_no_img)
    past_exists(save_dir_no_labels)

    file_img = os.listdir(path_img)
    file_lb = os.listdir(path_labels)
    pbar = tqdm(total=len(file_img), unit='progresso', dynamic_ncols=True, position=0, leave=True)
    for img, labels in zip(sorted(file_img), sorted(file_lb)):

        img_no = '{}split_{}_{}'.format(save_dir_no_img, 470, img.replace('jpg', 'png'))
        img_ex = '{}split_{}_{}'.format(save_dir_img, 470, img.replace('jpg', 'png'))
        #print(img_no, img_ex)
        #print( os.path.exists(img_ex), os.path.exists(img_no))
        if img.replace('.jpg', '') in labels and (not (os.path.exists(img_ex) or os.path.exists(img_no))):

            image = cv2.imread(path_img + img)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            split_width, split_height = size_crop

            clss, bbox_lb, data = load_labels(path_labels + labels, image.shape[:2])

            count = 0
            slices = calculate_slice_bboxes(img_h, img_w, split_height, split_width)

            for slice_img in slices:

                crop_transform = A.Compose(
                [A.Crop(*slice_img),],
                bbox_params=A.BboxParams(format="pascal_voc",
                                        label_fields=['labels'],
                                    min_visibility=0.1,
                                    min_area=0.1),
                )
                cropped = crop_transform(image=image, bboxes=bbox_lb, labels=clss)

                unique_values = np.unique(cropped['image'])
                if len(unique_values) > 1:#se imagem vazia deleta

                    crop_img = np.asarray(cropped['image'])
   
                    img_aux = Image.fromarray(crop_img)
               
                    img = img.replace('jpg','png')
                    labels = labels.replace('jpg','png')

                    if len(cropped['bboxes']) > 0:
                     
                        """with open('{}split_{}_{}'.format(save_dir_labels, count, labels), 'w') as f:
                            for x1,y1x2,y2 in cropped["bboxes"]:
                                f.write("{} {} {} {} {}\n".format(0,x1, y1, x2, y2))"""

                        with open('{}split_{}_{}'.format(save_dir_labels, count, labels), 'w') as f:
                            for x1,x2,y1,y2 in cropped["bboxes"]:
                                x,y,w,h = convert_voc_yolo(size_crop,[x1, y1, x2, y2])
                                f.write("{} {} {} {} {}\n".format(0,x,y,w,h))

           
                        img_aux.save('{}split_{}_{}'.format(save_dir_img, count, img))
                        
                    else:
                        
                        """with open('{}split_{}_{}'.format(save_dir_no_labels, count, labels), 'w') as f: #salvar coordenadas pascal voc
                            for x1,x2,y1,y2 in cropped["bboxes"]:
                                f.write("{} {} {} {} {}\n".format(0,x1, y1, x2, y2))"""

                        with open('{}split_{}_{}'.format(save_dir_no_labels, count, labels), 'w') as f: #salvar coordenadas yolo
                            for x1,x2,y1,y2 in cropped["bboxes"]:
                                x,y,w,h = convert_voc_yolo(size_crop,[x1, y1, x2, y2])
                                f.write("{} {} {} {} {}\n".format(0,x,y,w,h))

                    
                        #print('{}split_{}_{}'.format(save_dir_img, count, img))
                        img_aux.save('{}split_{}_{}'.format(save_dir_no_img, count, img))

            #show_image(cropped['image'], cropped['bboxes'], cropped['labels'])


                count += 1
        pbar.update(1)


    return 0

def convert_yolo(size_crop,path_labels):

    save_dir_labels = "yolo/train/"
    file_lb = os.listdir(path_labels)
    for labels in  sorted(file_lb):
        
        fl = open(path_labels + labels, 'r')
        data = fl.readlines()
        fl.close()

        print(labels)
        with open('{}{}'.format(save_dir_labels, labels), 'w') as f:
            for dt in data:
                cls, x,y,w,h = map(float, dt.split(' '))
                x,y,w,h = convert_voc_yolo(size_crop, [x,y,w,h])
                f.write("{} {} {} {} {}\n".format(0,x,y,w,h))


def  main():

    path_img = "../retinanet_pytorch/dataset_roi_tuberculosis/images/test/"
    path_lb = "../retinanet_pytorch/dataset_roi_tuberculosis/labels/test/"
   
    dest_img = "dataset_0.1/test/bacilo/images/"
    dest_lb = "dataset_0.1/test/bacilo/labels/"
    dest_no_lb = "dataset_0.1/test/no_bacilo/images/"
    dest_no_img = "dataset_0.1/test/no_bacilo/labels/"
    box_size = (640,640)

    crop_dataset(path_img, path_lb, box_size, dest_img, dest_lb, dest_no_img, dest_no_lb)
  
    #convert_yolo(box_size, "yolo/bacilo/train/labels/")

main()
