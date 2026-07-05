import os
import pandas as pd
import tensorflow as tf

class AIDataset:
    def __init__(self, image_dir=None, csv_file=None, root_dir=None, batch_size=32):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.image_paths = []
        self.labels = []

        if image_dir is not None:
            self._load_from_folder()
        elif csv_file is not None:
            self._load_from_csv()
        else:
            raise ValueError("Provide image_dir or csv_file")


    def _load_from_folder(self):
        for class_name in os.listdir(self.image_dir):
            class_path = os.path.join(self.image_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = 1 if class_name.lower() in ["real", "human", "real_image"] else 0
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def _load_from_csv(self):
        df = pd.read_csv(self.csv_file)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        if "file_name" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain file_name and label columns")

        base_path = self.root_dir if self.root_dir else os.path.dirname(self.csv_file)

        for _, row in df.iterrows():
            img_path = os.path.join(base_path, str(row["file_name"]))

            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels.append(int(row["label"]))


    def _load_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0  # normalize

        return image, label

    def get_dataset(self, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.labels)
        )
        dataset = dataset.map(
            self._load_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset