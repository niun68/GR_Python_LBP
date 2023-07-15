import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
import glob


def preprocess_image(images, trgt_size=(224, 224), dtype=np.uint8):
    lbp_features = []

    for img in images:
        if not isinstance(img, np.ndarray):
            try:
                img = np.array(img)
            except Exception as e:
                raise TypeError("The image data must be a NumPy array.") from e

        print("Image shape before preprocessing:", img.shape)

        if len(img.shape) != 3 or img.shape[2] != 3:
            if len(img.shape) == 2:
                # Convert grayscale image to suitable depth before converting to RGB
                if img.dtype == np.uint8:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError("The image data must be a 3-dimensional array "
                                 "with 3 channels. Convert the image to RGB.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        if height > width:
            pad_top = int((height - width) / 2)
            pad_bottom = height - width - pad_top
            img = cv2.copyMakeBorder(img, 0, 0, pad_top, pad_bottom, cv2.BORDER_CONSTANT)
        elif width > height:
            pad_left = int((width - height) / 2)
            pad_right = width - height - pad_left
            img = cv2.copyMakeBorder(img, pad_left, pad_right, 0, 0, cv2.BORDER_CONSTANT)
        img = cv2.resize(img, trgt_size)
        img = img.astype(dtype)

        # Convert the image to grayscale.
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate LBP features.
        lbp = local_binary_pattern(img_gray, 8, 2)

        lbp_features.append(lbp)

    return np.array(lbp_features)


def load_data(female_data_dir, male_data_dir):
    fmle_images = []
    mle_images = []

    for filepath in glob.glob(os.path.join(female_data_dir, "*.jpg"))[:1000]:
        img = cv2.imread(filepath)
        fmle_images.append(img)

    for filepath in glob.glob(os.path.join(male_data_dir, "*.jpg"))[:1000]:
        img = cv2.imread(filepath)
        mle_images.append(img)

    # Create a list of labels.
    t_labels = ["female"] * len(fmle_images) + ["male"] * len(mle_images)

    return fmle_images, mle_images, t_labels


def train_model(images, t_labels):
    # Convert the labels to a numerical format.
    encoder = LabelEncoder()
    t_labels = encoder.fit_transform(t_labels)

    features = []
    y = []

    for img, label in zip(images, t_labels):
        # Extract LBP features from the image.
        lbp = preprocess_image([img], (224, 224))  # Pass a single image as a list

        features.append(lbp.flatten())
        y.append(label)

    # Create and train the SVM model.
    clf = SVC(kernel="linear", probability=True)
    clf.fit(features, y)

    return clf


def test_model(calf, img):
    # Extract LBP features from the image.
    lbp = preprocess_image([img], (224, 224))  # Pass a single image as a list
    lbp = lbp.flatten()

    # Predict the gender of the image.
    pred = calf.predict([lbp])

    return pred[0]


def validate_model(calf, fmle_validation, mle_validation):
    correct = 0
    # total = 0

    for img in fmle_validation[:250]:
        lbp = preprocess_image([img], (224, 224))  # Pass a single image as a list
        lbp = lbp.flatten()

        pred = calf.predict([lbp])
        if pred[0] == 0:  # 0 represents "female" in the label encoding
            correct += 1

    for img in mle_validation[:250]:
        lbp = preprocess_image([img], (224, 224))  # Pass a single image as a list
        lbp = lbp.flatten()

        pred = calf.predict([lbp])
        if pred[0] == 1:  # 1 represents "male" in the label encoding
            correct += 1

    total = len(fmle_validation) + len(mle_validation)
    accuracy = correct / total

    print("Accuracy: {}%".format(accuracy * 100))


if __name__ == "__main__":
    female_images, male_images, _ = load_data("Images/Training/Female", "Images/Training/Male")
    female_validation, male_validation, _ = load_data("Images/Validation/female", "Images/Validation/male")
    labels = ["female"] * len(female_images) + ["male"] * len(male_images)
    model = train_model(preprocess_image(female_images + male_images), labels)
    validate_model(model, female_validation, male_validation)

    test_image = cv2.imread("profilo_rid.jpg")

    prediction = test_model(model, test_image)

    if prediction == 0:
        print("The predicted gender is: female")
    elif prediction == 1:
        print("The predicted gender is: male")
