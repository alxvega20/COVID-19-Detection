import os
import cv2
import numpy as np
import pandas as pd

def load_data(base_dir, metadata_csv, img_size=224):
    df = pd.read_csv(metadata_csv)
    df = df[df["Label"].isin(["Normal", "Pnemonia"])]
    label_map = {"Normal": 0, "Pnemonia": 1}
    df["Target"] = df["Label"].map(label_map)

    def load_images(dataframe, folder_name):
        images, labels = [], []
        for _, row in dataframe.iterrows():
            img_path = os.path.join(base_dir, folder_name, row["X_ray_image_name"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                images.append(img)
                labels.append(row["Target"])
        return np.array(images), np.array(labels)

    train_df = df[df["Dataset_type"] == "TRAIN"]
    test_df = df[df["Dataset_type"] == "TEST"]

    X_train, y_train = load_images(train_df, "train")
    X_test, y_test = load_images(test_df, "test")

    X_train = X_train.reshape(-1, img_size, img_size, 1)
    X_test = X_test.reshape(-1, img_size, img_size, 1)


    ####debugging
    print(f"Train CSV rows: {len(train_df)}")
    print(f"Test CSV rows: {len(test_df)}")
    print(df["Label"].value_counts())
    ####

    return X_train, y_train, X_test, y_test
