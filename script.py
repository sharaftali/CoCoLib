from coco_lib.common import Info, Image as coco_image, License
from coco_lib.objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory, ObjectDetectionDataset

categories = [ObjectDetectionCategory(
    id=1,
    name='article',
    supercategory=''
), ObjectDetectionCategory(
    id=2,
    name='author',
    supercategory='article'
), ObjectDetectionCategory(
    id=3,
    name='column',
    supercategory='article'
), ObjectDetectionCategory(
    id=4,
    name='title',
    supercategory='article'
)]

for cat in categories:
    print(cat.id)
    print(cat.name)
    print(cat.__dict__)
