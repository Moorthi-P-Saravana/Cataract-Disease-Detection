import cv2
import os
import albumentations as A


# Load an image
def load_image(image_path):
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image


# Define augmentation functions
def augment_image(image, augmentation):
    augmented = augmentation(image=image)
    return augmented['image']


# Apply augmentations and save them
def apply_augmentations(image_path, save_folder):
    image = load_image(image_path)  # Load original image
    folder_to_save = image_path.split('/')[1].split('_')[1]
    os.makedirs(f'{save_folder}/{folder_to_save}', exist_ok=True)

    # Define 7 augmentation techniques
    augmentations = [
        ("rotated", A.Rotate(limit=30, p=1)),  # Rotation ±30°
        ("horizontal_flip", A.HorizontalFlip(p=1)),  # Horizontal flip
        ("vertical_flip", A.VerticalFlip(p=1)),  # Vertical flip
        ("brightness", A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1)),  # Brightness change
        ("gaussian_noise", A.GaussNoise(var_limit=(10.0, 50.0), p=1)),  # Gaussian noise
        ("elastic_transform", A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1)),  # Elastic distortion
        ("resized", A.Resize(384, 512, p=1))  # Resize to 256x256
    ]

    # Save the original image
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(save_folder, f"{folder_to_save}", f"{base_filename}_original.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Apply each augmentation and save the output
    for aug_name, aug in augmentations:
        augmented_image = augment_image(image, aug)
        save_path = os.path.join(save_folder, f"{folder_to_save}", f"{base_filename}_{aug_name}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))


def Augmentation():
    Label_data = os.listdir('Dataset')
    for label in Label_data:
        images = os.listdir(f'Dataset/{label}')
        for img in images:
            image_path = f'Dataset/{label}/{img}'
            apply_augmentations(image_path, save_folder='augmented_images')
