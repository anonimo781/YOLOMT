from __future__ import annotations

import json
import os


def check_image_ann(
    path_img: str,
    file_format: str,
    show_file_empty: bool = True,
):
    cont = 0

    print(path_img, flush=True)
    for img in os.listdir(path_img):

        file_ann = img.replace(
            'images', file_format,
        ).replace('jpg', file_format)

        if not os.path.exists(file_ann):
            cont += 1
            if show_file_empty:
                print(file_ann)

            else:
                break

    if cont == 0:
        print('Annotations OK!!')
    else:
        print('Annotations empty!!')


def move_img_ann(
    path_img: str,
    path_ann: str,
    file_format_img: str,
    file_format_ann: str,
    path_dest: str,
):

    cont = 0
    for arq in os.listdir(path_ann):

        arq = arq.replace(file_format_ann, file_format_img).split('/')[-1]

        img = path_img + arq

        if os.path.exists(img):

            cont += 1
            path_dest += arq
        else:
            print(img)

            # shutil.copy(img,  path_dest)

    print(f'Foram movidos {cont} arquivo(s)')


def modify_json(
    path_json: str,
    tags_cls: dict,
):
    with open(path_json) as file:
        data = json.load(file)
        aux = []
        for ann in data['annotations']:

            ann['category_id'] = tags_cls[ann['category_id']]

            aux.append(ann)

        data['annotaions'] = aux

    data = json.dumps(data, indent=4)

    with open(path_json, 'w') as file:

        # write
        file.write(data)

    print('Json Modificado')


def main():  # pragma: no cover
    """path_img = r"base_ciaten/images/train" file_format= "xml".

    check_image_ann(path_img,file_format)

    ori_ann = r'base_ciaten/xml/train' ori_img = 'annot/' dest = r'base_ciaten/images/train/'

    # move_img_ann(ori_img, ori_ann,"jpg","xml",dest)
    """

    json = 'base_ciaten/coco/test/_annotations.coco.json'

    dicc = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    modify_json(json, dicc)


if __name__ == '__main__':
    raise SystemExit(main())
