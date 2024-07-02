import os
import shutil
from tqdm import tqdm
import random
import cv2
from PIL import Image
import numpy as np

def _read_arq(name):

    with open(name, 'r') as arquivo:
        # Lê o conteúdo do arquivo
        conteudo = arquivo.read()

        # Verifica se o conteúdo está vazio
        if not conteudo:
            return 0
        else:
            return 1

def mover_txt_img(path_lb, path_img, dest_img, dest_lb, dest_no_img, dest_no_lb):
    
    files_lb = os.listdir(path_lb)
    print(len(files_lb))
    files_img = os.listdir(path_img)
    files_img = random.sample(files_img, 10000)
    pbar = tqdm(total=len(files_img), unit='progresso', dynamic_ncols=True, position=0, leave=True)

    
    
    with open('{}.txt'.format("valid"), 'w') as f:

        for img   in files_img:
            
            image = cv2.imread(path_img + img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image)
            image = Image.fromarray(image)
            result  = True #_read_arq(path_lb + label)
            
            if result: 
                
                image.save( dest_img + img)
                #shutil.copy(path_img + img, dest_img + img)
                shutil.copy(path_lb + img[:-3] + "txt", dest_lb + img[:-3] + "txt")
                
            else:
                continue
                #unique_values = np.unique(image)
                #if len(unique_values) > 1:

                    #image.save( dest_no_img + img)
                #shutil.copy(path_lb + label,  dest_no_lb + label)
                   
                    

            pbar.update(1)


# Exemplo de uso
origin_img = "dataset_mlp_200x200/bacilo/train/images/"
origin_lb = "dataset_mlp_200x200/bacilo/train/labels/"



dest_img = "yolo_200x200/train/images/"
dest_lb = "yolo_200x200/train/labels/"
dest_no_lb = "dataset_mlp_200x200/no_bacilo/test/labels/"
dest_no_img = "dataset_mlp_200x200/no_bacilo/test/images/"

mover_txt_img(origin_lb, origin_img, dest_img, dest_lb, dest_no_img, dest_no_lb)
