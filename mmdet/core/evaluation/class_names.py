import mmcv


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
        # 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        # 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        # 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        'airplane', 'airport', 'baseball field', 'basketball court', 'bridge',
        'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golf field',
        'ground track field', 'harbor', 'ship', 'stadium', 'storage tank',
        'tennis court', 'train station', 'vehicle', 'windmill', 'overpass'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
        # 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        # 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        # 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        # 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        # 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        # 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        # 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        # 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        # 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        # 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        # 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        # 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'(coco)
        # 'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
        # 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
        # 'roundabout', 'soccer ball field', 'swimming-pool', 'container-crane'
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
        'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
        'harbor', 'swimming-pool', 'helicopter', 'container-crane'
        # "categories":[
        # {"supercategory": "none", "id": 1, "name": "plane"},
        # {"supercategory": "none", "id": 2, "name": "baseball-diamond"},
        # {"supercategory": "none", "id": 3, "name": "bridge"},
        # {"supercategory": "none", "id": 4, "name": "ground-track-field"},
        # {"supercategory": "none", "id": 5, "name": "small-vehicle"},
        # {"supercategory": "none", "id": 6, "name": "large-vehicle"},
        # {"supercategory": "none", "id": 7, "name": "ship"},
        # {"supercategory": "none", "id": 8, "name": "tennis-court"},
        # {"supercategory": "none", "id": 9, "name": "basketball-court"},
        # {"supercategory": "none", "id": 10, "name": "storage-tank"},
        # {"supercategory": "none", "id": 11, "name": "soccer-ball-field"},
        # {"supercategory": "none", "id": 12, "name": "roundabout"},
        # {"supercategory": "none", "id": 13, "name": "harbor"},
        # {"supercategory": "none", "id": 14, "name": "swimming-pool"},
        # {"supercategory": "none", "id": 15, "name": "helicopter"},
        # {"supercategory": "none", "id": 16, "name": "container-crane"}]}
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
