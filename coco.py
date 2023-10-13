import json
from datetime import datetime
from typing import List, Dict, Tuple
import requests
from core.logger import log
from coco_lib.common import Info, Image as coco_image, License
from coco_lib.objectdetection import ObjectDetectionAnnotation, ObjectDetectionCategory, ObjectDetectionDataset
from config import PROTECTED_BUCKET_NAME, SEGMENTED_BUCKET_NAME, MINIO_ADDRESS, CONTENT_BUCKET_NAME, OBJECT_API

categories = [ObjectDetectionCategory(
    id=1,
    name='content',
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
    name='content_title',
    supercategory='article'
)]


class CustomObjectDetectionAnnotation(ObjectDetectionAnnotation):
    def __init__(self, id: int, image_id: int, category_id: int, segmentation: List[List[float]], area: float,
                 bbox: Tuple[float, float, float, float], iscrowd: int, attributes: Dict):
        super().__init__(id, image_id, category_id, segmentation, area, bbox, iscrowd)
        self.attributes = attributes


def fill_coco_images(articles: List) -> List:
    images = []
    image_index = 1
    for index, article in enumerate(articles, start=1):
        for item in [article] + article['authors'] + article['columns'] + article['titles']:
            item['image_id'] = image_index
            images.append(coco_image(
                id=image_index,
                width=item['bbox']['width'],
                height=item['bbox']['height'],
                file_name=item.get('minio_img_address'),
                license=0,
                flickr_url='',
                coco_url='',
                date_captured=None
            ))
            image_index += 1
    return images


def fill_coco_annotations(articles: List):
    index = 1
    annotations = []
    for article in articles:
        for item in [article] + article['authors'] + article['columns'] + article['titles']:
            category_id = None
            bucket_name = SEGMENTED_BUCKET_NAME
            for cat in categories:
                if item['bbox']['label'] == cat.name:
                    category_id = cat.id

            if item['bbox']['label'] == "content":
                bucket_name = PROTECTED_BUCKET_NAME

            annotations.append(CustomObjectDetectionAnnotation(
                id=index,
                image_id=item['image_id'],
                category_id=category_id,
                segmentation=[],
                area=cal_area(item),
                bbox=bbox_dict_to_tuple(item['bbox']),
                iscrowd=0,
                attributes={
                    "URL": f"{MINIO_ADDRESS}/{bucket_name}/{item.get('minio_img_address')}",
                    "occluded": False,
                    "rotation": 0.0
                }
            ))
            index += 1
    return annotations


def bbox_dict_to_tuple(data: Dict) -> Tuple:
    if data.get('label'):
        del data['label']
    return data["x_min"], data["y_min"], data["x_max"], data["y_max"], data["width"], data["height"]


def cal_area(data):
    return data["bbox"]["width"] * data["bbox"]["height"]


def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()


def upload_documents_json_at_minio(data, custom_name):
    resp = requests.post(
        OBJECT_API, params={
            'bucket_name': CONTENT_BUCKET_NAME,
            "custom_name": custom_name,
            "unique_minio_address": False
        },
        files={'file': ('document.json', data)}
    )
    log.debug(resp)
    if resp.status_code in {200, 201, 202}:
        log.debug(f"Json is created for the page: {custom_name} and resp{resp.json()} ")
        return resp
    else:
        log.debug("Json is not created!")
        log.debug(resp.json())
        return None


def fill_coco_file(articles: List, custom_name):
    log.debug("fill coco file function called")
    filled_images = fill_coco_images(articles)
    log.debug("images filled")
    annotations = fill_coco_annotations(articles)
    log.debug("annotations........")
    info = Info(
        year=0,
        version="",
        description="",
        contributor="",
        url="",
        date_created=datetime.now())

    licenses = [License(
        id=0,
        name="",
        url=""
    )]
    coco_data = ObjectDetectionDataset(
        annotations=annotations,
        categories=categories,
        images=filled_images,
        info=info,
        licenses=licenses
    )

    coco_data_dict = {
        "info": coco_data.info.__dict__,
        "images": [image.__dict__ for image in coco_data.images],
        "licenses": [license.__dict__ for license in coco_data.licenses],
        "categories": [category.__dict__ for category in coco_data.categories],
        "annotations": [annotation.__dict__ for annotation in coco_data.annotations],
    }
    json_data = json.dumps(coco_data_dict, indent=4, default=datetime_serializer, ensure_ascii=False)

    upload_documents_json_at_minio(json_data, custom_name)
