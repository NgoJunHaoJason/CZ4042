import os


root_dir_path = '/home/burntice/3_data/'

source_dir_path = os.path.join(root_dir_path, 'Adience/')
destination_dir_path = os.path.join(root_dir_path, 'Adience_processed/')

num_folds = 5

# key: image_dir + '-' + image_name
image_label_map = {}

# build up a mapping, from image to its corresponding label (m/f)
for fold_index in range(num_folds):
    fold_text = f'fold_{fold_index}_data.txt'
    fold_text_path = os.path.join(source_dir_path, fold_text)

    with open(fold_text_path) as text_file:
        text_file.readline()  # get rid of header line

        for line in text_file:
            datum_features = line.split('\t')

            gender = datum_features[4]
            if gender not in ('m', 'f'):
                continue  # invalid gender

            age = datum_features[3]
            if age == ' None':
                continue  # invalid age

            label = gender + '-' + age  # label is gender-age pair

            image_dir = datum_features[0]
            image_name = datum_features[1]

            map_key = image_dir + '-' + image_name

            image_label_map[map_key] = label

print('built mapping of images to their corresponding labels')

faces_dir_path = os.path.join(source_dir_path, 'faces/')

for image_dir in os.listdir(faces_dir_path):
    image_dir_path = os.path.join(faces_dir_path, image_dir)

    for image in os.listdir(image_dir_path):
        if not image.endswith('.jpg'):  # not actually an image
            continue

        image_path = os.path.join(image_dir_path, image)

        image_name = image.split('.')[2] + '.jpg'

        map_key = image_dir + '-' + image_name
        
        if map_key not in image_label_map:
            print(f'bad datum: {map_key}')
            continue

        label = image_label_map[map_key]  # serves as new dir for images
        label_path = os.path.join()

        new_image_path = os.path.join(destination_dir_path, label, map_key)
        print(new_image_path)
        break

    break
