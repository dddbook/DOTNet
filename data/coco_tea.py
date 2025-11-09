dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='tea_point1', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='tea_point2',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='tea_point3',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='top_1',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        4:
        dict(
            name='top_2',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        5:
        dict(
            name='top_3',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='b_2',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        7:
        dict(
            name='b_3',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap=''),

    },
    skeleton_info={
        0:
        dict(link=('top_1', 'tea_point1'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('tea_point1', 'b_2'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('b_2', 'tea_point2'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('tea_point2', 'b_3'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('b_3', 'tea_point3'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('top_2', 'b_2'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('top_3', 'b_3'), id=6, color=[255, 128, 0]),
    },
    joint_weights=[
        1.5, 1.5, 1.5, 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        0.107, 0.107, 0.107, 0.107, 0.107,
        0.062, 0.107, 0.107
    ])
