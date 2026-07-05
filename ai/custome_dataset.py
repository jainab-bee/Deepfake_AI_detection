import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import DATASET_DIR, TRAIN_CSV, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_STATE
from preprocessing import load_and_preprocess, augment_image


def load_dataframes(use_subset=True, samples_per_class=5000):
    df = pd.read_csv(TRAIN_CSV)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if use_subset:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(samples_per_class, random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )

    df["file_name"] = df["file_name"].apply(lambda x: str(DATASET_DIR / x))

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    return train_df, val_df


def make_dataset(df, training=True):
    image_paths = df["file_name"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset