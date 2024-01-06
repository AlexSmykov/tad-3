import cv2
import os
import dvc.api

VARIANTS = ['food', 'non_food']
IMAGE_SIZE = (256, 256)

dvc_params = dvc.api.params_show()

def getProcessedImage(path):
    pic = cv2.imread(path)
    pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    
    return cv2.resize(pic_gray, IMAGE_SIZE)

def saveWithAugment(path, img):
    cv2.imwrite(path + '.jpg', img)
    if (dvc_params['with_augmentation']):
        cv2.imwrite(path + '_augmented-1.jpg', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        cv2.imwrite(path + '_augmented-2.jpg', cv2.rotate(img, cv2.ROTATE_180))
        cv2.imwrite(path + '_augmented-3.jpg', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

print('Предобрабокта данных')
if (dvc_params['with_augmentation']):
    print('С аугментацией')

try:
    os.mkdir('dataset')
except:
    pass

try:
    for variant in VARIANTS:
        os.mkdir(f'dataset/{variant}')
except:
    pass

kaggle_ds = "Kaggle_ds"

i_1 = 0
i_2 = 0
for dir in os.listdir(kaggle_ds):
    for dir_class in os.listdir(f'{kaggle_ds}/{dir}'):
        print(f'{kaggle_ds}/{dir}/{dir_class}')
        if dir_class == "food":
            for pic_path in os.listdir(f'{kaggle_ds}/{dir}/{dir_class}'):
                processed_image = getProcessedImage(f'{kaggle_ds}/{dir}/{dir_class}/{pic_path}')
                saveWithAugment(f'dataset/{VARIANTS[0]}/{i_1}', processed_image)
                i_1 += 1

        else :
            for pic_path in os.listdir(f'{kaggle_ds}/{dir}/{dir_class}'):
                processed_image = getProcessedImage(f'{kaggle_ds}/{dir}/{dir_class}/{pic_path}')
                saveWithAugment(f'dataset/{VARIANTS[1]}/{i_2}', processed_image)
                i_2 += 1