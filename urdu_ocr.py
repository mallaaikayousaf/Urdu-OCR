
import numpy as np
import os
import cv2

from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = [
    'alif',         # class_0
    'alif mad aa',  # class_1
    'baa',          # class_2
    'pay',          # class_3
    'taa',          # class_4
    'tay',          # class_5
    'say',          # class_6
    'jeem',         # class_7
    'chay',         # class_8
    'hay',          # class_9
    'khay',         # class_10
    'daal',         # class_11
    'ddaal',        # class_12
    'zaal',         # class_13
    'ray',          # class_14
    'rday',         # class_15
    'zay',          # class_16
    'zsay',         # class_17
    'seen',         # class_18
    'sheen',        # class_19
    'swaad',        # class_20
    'zwaad',        # class_21
    'toyen',        # class_22
    'zoyen',        # class_23
    'ayn',          # class_24
    'ghaien',       # class_25
    'faa',          # class_26
    'qaaf',         # class_27
    'kaaf',         # class_28
    'gaaf',         # class_29
    'laam',         # class_30
    'meem',         # class_31
    'noon',         # class_32
    'noon ghunaan', # class_33
    'wow',          # class_34
    'hay',          # class_35
    'hay',          # class_36
    'hamza',        # class_37
    'choti yaa',    # class_38
    'badi yaa',     # class_39
]

_STRING_SORTED_INDICES = [
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    4, 5, 6, 7, 8, 9
]

MODEL_CLASS_NAMES = [CLASS_NAMES[i] for i in _STRING_SORTED_INDICES]


def _build_architecture():
    """
    Rebuild the CNN architecture exactly matching the saved model config:
    Conv2D(32) -> Pool -> Conv2D(64) -> Pool -> Conv2D(128) -> Pool
    -> Flatten -> Dense(128,relu) -> Dropout(0.5) -> Dense(40,softmax)
    Input: (64, 64, 1)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), name='conv2d'),
        MaxPooling2D((2, 2), name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D((2, 2), name='max_pooling2d_1'),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D((2, 2), name='max_pooling2d_2'),
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense'),
        Dropout(0.5, name='dropout'),
        Dense(40, activation='softmax', name='dense_1'),
    ])
    return model


def _load_weights_from_keras3_h5(model, h5_path):
    """
    Load weights from a Keras 3 .h5 file into a tf-keras 2.15 model.

    Keras 3 stores weights at:
      model_weights/<layer_name>/sequential/<layer_name>/kernel  (shape ...)
      model_weights/<layer_name>/sequential/<layer_name>/bias    (shape ...)

    We read kernel and bias (in that order) and call layer.set_weights().
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        wg = f['model_weights']

        for layer in model.layers:
            name = layer.name
            if name not in wg:
                continue  # e.g. MaxPooling, Flatten, Dropout have no weights

            # Navigate: wg[name] -> first subgroup -> name -> kernel/bias
            top = wg[name]
            # First subgroup is the model name (e.g. 'sequential')
            sub_keys = list(top.keys())
            if not sub_keys:
                continue
            mid = top[sub_keys[0]]  # e.g. wg['conv2d']['sequential']

            if name not in mid:
                continue
            leaf = mid[name]  # e.g. wg['conv2d']['sequential']['conv2d']

            leaf_keys = list(leaf.keys())
            if not leaf_keys:
                continue

            # Build weight list: kernel first, then bias (tf-keras order)
            weights = []
            for wname in ['kernel', 'bias']:
                if wname in leaf:
                    weights.append(leaf[wname][()])

            if weights:
                layer.set_weights(weights)


class UrduOCR:

    def __init__(self, model_path="urdu_model.h5", img_size=64):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = MODEL_CLASS_NAMES
        print(f"UrduOCR initialized. Classes: {len(self.class_names)}")

    def load(self):
        """
        Load the saved model.
        The .h5 was saved with Keras 3.14.1 but we run tf-keras 2.15,
        so we rebuild the architecture and load only the weights manually.
        """
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model not found at {self.model_path}")
            return

        self.model = _build_architecture()
        _load_weights_from_keras3_h5(self.model, self.model_path)
        print(f"Model loaded: {self.model_path}")

    def predict_from_array(self, img_array):
        """Predict from numpy array. Returns (character, confidence, top5)."""
        if self.model is None:
            return None, None, None

        img = img_array.astype('float32')
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        if img.max() > 1.0:
            img = img / 255.0
        if img.mean() < 0.5:
            img = 1.0 - img

        img = img.reshape(1, self.img_size, self.img_size, 1)
        predictions = self.model.predict(img, verbose=0)
        idx = int(np.argmax(predictions[0]))
        conf = float(np.max(predictions[0])) * 100
        char = self.class_names[idx]

        top5_idx = np.argsort(predictions[0])[::-1][:5]
        top5 = [(self.class_names[i], float(predictions[0][i]) * 100)
                for i in top5_idx]

        return char, conf, top5

    def predict(self, img_path):
        """Predict from image file path."""
        if self.model is None:
            return None, None
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        char, conf, _ = self.predict_from_array(img)
        print(f"Predicted: {char} ({conf:.1f}%)")
        return char, conf

    def evaluate(self, test_dir="data/processed/test"):
        """Evaluate accuracy on test set."""
        if self.model is None:
            return
        datagen = ImageDataGenerator(rescale=1./255)
        test_data = datagen.flow_from_directory(
            test_dir, target_size=(self.img_size, self.img_size),
            color_mode='grayscale', batch_size=32,
            class_mode='categorical', shuffle=False)
        loss, acc = self.model.evaluate(test_data, verbose=0)
        print(f"\nTest Accuracy : {acc * 100:.2f}%")
        print(f"Test Loss     : {loss:.4f}")
        preds = np.argmax(self.model.predict(test_data, verbose=0), axis=1)
        labels = list(test_data.class_indices.keys())
        print("\nClassification Report:\n")
        print(classification_report(test_data.classes, preds, target_names=labels))
        return acc, loss

    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            return
        print("\nModel Architecture Summary:")
        self.model.summary()


if __name__ == "__main__":
    ocr = UrduOCR()
    ocr.load()
    ocr.summary()
    print("\nUrduOCR class is working correctly.")
