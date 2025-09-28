import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
from skimage.feature import local_binary_pattern
import os
from save_load import *
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def compute_gdp2_features(image):
    """
    Computes the second-order Gradient-Based Differential Pattern (GDP2).
    """
    radius = 2
    points = 8 * radius  # More neighborhood points than GDP
    lbp = local_binary_pattern(image, points, radius, method='uniform')
    return lbp


def compute_ldipv_features(image):
    """
    Computes Local Directional Pattern Variance (LDiPv).
    Uses Sobel gradient filtering to capture directional variations.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Convert to degrees

    # Normalize direction values to 8 bins (0-360 degrees mapped to 8 bins)
    direction_bins = np.digitize(gradient_direction, bins=np.linspace(-180, 180, 9)) - 1

    ldipv_feat = gradient_magnitude * direction_bins  # LDiPv encodes variance of directional gradients

    return ldipv_feat


def compute_ldip_features(image):
    """
    Computes Local Directional Pattern (LDiP).
    Uses Prewitt operator to capture directional features.
    """
    prewitt_x = cv2.filter2D(image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    gradient_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    gradient_direction = np.arctan2(prewitt_y, prewitt_x) * (180 / np.pi)  # Convert to degrees

    # Normalize to 8 orientation bins
    orientation_bins = np.digitize(gradient_direction, bins=np.linspace(-180, 180, 9)) - 1

    return orientation_bins


def compute_gdp_features(image):
    """
    Extracts Gradient-Based Differential Pattern (GDP) features using Local Binary Patterns (LBP).
    """
    radius = 1
    points = 8 * radius  # Neighborhood points
    lbp = local_binary_pattern(image, points, radius, method='uniform')
    return lbp


def raf_filter_rgb(image, num_iter=15, kappa=30, gamma=0.1, option=1):
    """
    Applies the Regularized Anisotropic Filtering (RAF) to an RGB image.

    Parameters:
        image (numpy.ndarray): Input RGB image.
        num_iter (int): Number of iterations for diffusion.
        kappa (float): Conductance parameter (controls sensitivity to edges).
        gamma (float): Step size (should be <= 0.25 for stability).
        option (int): Type of diffusion function (1: Exponential, 2: Quadratic).

    Returns:
        numpy.ndarray: Denoised RGB image.
    """
    def anisotropic_diffusion(img_channel):
        """
        Applies anisotropic diffusion to a single grayscale channel.
        """
        img_channel = img_channel.astype(np.float32)  # Convert to float
        img_filtered = img_channel.copy()

        for _ in range(num_iter):
            img_pad = np.pad(img_filtered, ((1, 1), (1, 1)), mode='reflect')

            # Compute gradients in 4 directions
            dN = img_pad[:-2, 1:-1] - img_filtered  # North
            dS = img_pad[2:, 1:-1] - img_filtered   # South
            dE = img_pad[1:-1, 2:] - img_filtered   # East
            dW = img_pad[1:-1, :-2] - img_filtered  # West

            # Compute diffusion function
            if option == 1:
                cN = np.exp(-(dN / kappa) ** 2)
                cS = np.exp(-(dS / kappa) ** 2)
                cE = np.exp(-(dE / kappa) ** 2)
                cW = np.exp(-(dW / kappa) ** 2)
            elif option == 2:
                cN = 1 / (1 + (dN / kappa) ** 2)
                cS = 1 / (1 + (dS / kappa) ** 2)
                cE = 1 / (1 + (dE / kappa) ** 2)
                cW = 1 / (1 + (dW / kappa) ** 2)

            # Update the image using the PDE
            img_filtered += gamma * (cN * dN + cS * dS + cE * dE + cW * dW)

        return np.clip(img_filtered, 0, 255).astype(np.uint8)  # Convert back to uint8

    # Split the RGB image into separate channels
    r, g, b = cv2.split(image)

    # Apply RAF filter to each channel
    r_filtered = anisotropic_diffusion(r)
    g_filtered = anisotropic_diffusion(g)
    b_filtered = anisotropic_diffusion(b)

    # Merge back the channels
    denoised_image = cv2.merge([r_filtered, g_filtered, b_filtered])
    return denoised_image


def preprocess_image(image_path):
    """
    Loads an image, applies anisotropic filtering, and extracts GDP, GDP2, LDiPv, and LDiP features.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image Denoising
    denoised_image = raf_filter_rgb(image, num_iter=15, kappa=30, gamma=0.1, option=1)

    # Image enhancement
    # Apply histogram equalization on each channel separately
    r, g, b = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # Merge back the equalized channels
    hist_eq_image = cv2.merge([r_eq, g_eq, b_eq])
    #
    # # Create a figure with two subplots
    # plt.figure(figsize=(8, 4))
    #
    # # Show Original Image
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)
    # plt.title("Original Image", fontweight="bold", fontfamily="serif", fontsize=14)
    # plt.axis("off")  # Hide axes
    #
    # # Show Denoised Image
    # plt.subplot(1, 3, 2)
    # plt.imshow(denoised_image)
    # plt.title("Denoised Image", fontweight="bold", fontfamily="serif", fontsize=14)
    # plt.axis("off")  # Hide axes
    #
    # # Show Enhanced Image
    # plt.subplot(1, 3, 3)
    # plt.imshow(hist_eq_image)
    # plt.title("Enhanced Image", fontweight="bold", fontfamily="serif", fontsize=14)
    # plt.axis("off")  # Hide axes
    #
    # # Adjust layout and display
    # plt.tight_layout()
    # plt.savefig('Data Visualization/cataract.png')
    # plt.show()

    image = cv2.resize(image, (128, 128))
    denoised_image = cv2.resize(denoised_image, (128, 128))

    original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised_gray = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)

    # Compute metrics
    mse_value = mse(original_gray, denoised_gray)
    psnr_value = psnr(original_gray, denoised_gray, data_range=255)

    # For comparing with existing techniques
    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    gaussian_gray = cv2.cvtColor(gaussian_blur, cv2.COLOR_RGB2GRAY)
    gaussian_mse_value = mse(original_gray, gaussian_gray)
    gaussian_psnr_value = psnr(original_gray, gaussian_gray, data_range=255)

    # Apply Median Blur
    median_blur = cv2.medianBlur(image, 5)
    median_gray = cv2.cvtColor(median_blur, cv2.COLOR_RGB2GRAY)
    median_mse_value = mse(original_gray, median_gray)
    median_psnr_value = psnr(original_gray, median_gray, data_range=255)

    # Apply Bilateral Filter
    bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
    bilateral_gray = cv2.cvtColor(bilateral_filter, cv2.COLOR_RGB2GRAY)
    bilateral_mse_value = mse(original_gray, bilateral_gray)
    bilateral_psnr_value = psnr(original_gray, bilateral_gray, data_range=255)

    # Apply Non-Local Means Denoising
    nlm_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    nlm_gray = cv2.cvtColor(nlm_denoised, cv2.COLOR_RGB2GRAY)
    nlm_mse_value = mse(original_gray, nlm_gray)
    nlm_psnr_value = psnr(original_gray, nlm_gray, data_range=255)

    # Define technique names
    techniques = ["Proposed", "Gaussian Blur", "Median Blur", "Bilateral Filter", "Non-Local Means"]
    # MSE values for each technique
    mse_values = [mse_value, gaussian_mse_value, median_mse_value, bilateral_mse_value, nlm_mse_value]
    # PSNR values for each technique
    psnr_values = [psnr_value, gaussian_psnr_value, median_psnr_value, bilateral_psnr_value, nlm_psnr_value]


    # # Set figure size
    # plt.figure(figsize=(10, 6))
    # # MSE Bar Chart
    # plt.subplot(1, 2, 1)
    # plt.bar(techniques, mse_values, color=['red', 'blue', 'green', 'orange', 'purple'])
    # plt.xlabel("Denoising Techniques", fontweight="bold", fontfamily="serif", fontsize=12)
    # plt.ylabel("MSE", fontweight="bold", fontfamily="serif", fontsize=12)
    # plt.title("Mean Squared Error (MSE)", fontweight="bold", fontfamily="serif", fontsize=14)
    # plt.xticks(rotation=90, fontweight="bold", fontfamily="serif")
    # plt.yticks(fontweight="bold", fontfamily="serif")
    # # PSNR Bar Chart
    # plt.subplot(1, 2, 2)
    # plt.bar(techniques, psnr_values, color=['red', 'blue', 'green', 'orange', 'purple'])
    # plt.xlabel("Denoising Techniques", fontweight="bold", fontfamily="serif", fontsize=12)
    # plt.ylabel("PSNR", fontweight="bold", fontfamily="serif", fontsize=12)
    # plt.title("Peak Signal-to-Noise Ratio (PSNR)", fontweight="bold", fontfamily="serif", fontsize=14)
    # plt.xticks(rotation=90, fontweight="bold", fontfamily="serif")
    # plt.yticks(fontweight="bold", fontfamily="serif")
    # # Show plots
    # plt.tight_layout()
    # plt.savefig('Data Visualization/MSE-PSNR plot.png')
    # plt.show()

    enhanced_image = cv2.resize(hist_eq_image, (128, 128))

    return enhanced_image


def feature_extraction(enhanced_image):

    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Feature extraction
    gdp = compute_gdp_features(enhanced_image)
    gdp2 = compute_gdp2_features(enhanced_image)
    ldipv = compute_ldipv_features(enhanced_image)
    ldip = compute_ldip_features(enhanced_image)

    # Stack features as multi-channel input (treat features as separate image channels)
    feature_stack = np.stack([gdp, gdp2, ldipv, ldip], axis=-1)

    return feature_stack


def datagen():
    labels = []
    features = []
    Label_data = os.listdir('augmented_images')
    for label in Label_data:
        images = os.listdir(f'augmented_images/{label}')
        for img in images:
            image_path = f'augmented_images/{label}/{img}'

            # Preprocessing
            preprocessed_image = preprocess_image(image_path)

            # Feature Extraction
            feature = feature_extraction(preprocessed_image)
            features.append(feature)
            labels.append(label)

    features = np.array(features)

    save('Features', features)
    save('Labels', labels)

    lab_encoder = LabelEncoder()
    labels = lab_encoder.fit_transform(labels)
    joblib.dump(lab_encoder, 'Saved Data/label encoder.joblib')

    features = features.astype("float32") / np.max(features)  # Normalize

    # Train-Test Split
    train_sizes = [0.7, 0.8]
    for train_size in train_sizes:

        x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_size, random_state=42)

        save('x_train_' + str(int(train_size * 100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)

