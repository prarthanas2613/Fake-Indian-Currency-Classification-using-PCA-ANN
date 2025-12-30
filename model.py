import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob  # Import glob

# ------------------------------ Configuration ------------------------------
BASE_DIR = os.getenv("DATASET_DIR", r"C:\Users\HP\.cache\kagglehub\datasets\gauravsahani\indian-currency-notes-classifier\versions\1")
OUTPUT_DIR = "data"
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "Preprocessed")
META_DIR = os.path.join(OUTPUT_DIR, "Meta")
MODEL_DIR = "models"

IMG_WIDTH, IMG_HEIGHT = 128, 128
NUM_CHANNELS = 3  # Color images

CATEGORIES = ["1Hundrednote", "2Hundrednote", "2Thousandnote", "5Hundrednote", "Fiftynote", "Tennote", "Twentynote"]
TRAIN_DIR_NAME = "Train"
TEST_DIR_NAME = "Test"

NUM_CLASSES = len(CATEGORIES)  # Number of classes for classification

# Ensure directories exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------ Helper Functions ------------------------------
def load_image(image_path):
    """Loads an image, resizes it, and normalizes it. Returns None if the image cannot be loaded."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize image
        img = img / 255.0  # Normalize pixel values to [0, 1]
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_dataset(base_dir, categories, output_dir_name, image_paths):
    """Preprocesses images and saves them as .npy files."""
    output_dir = os.path.join(PREPROCESSED_DIR, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    processed_image_paths = []  # To store paths of successfully processed images

    for img_path in tqdm(image_paths, desc=f"Preprocessing {output_dir_name}"):
        try:
            img = load_image(img_path)
            if img is not None:
                output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.npy')
                np.save(output_path, img)
                processed_image_paths.append(img_path)  # Only append path if processed
        except Exception as e:
            print(f"Error during preprocessing of {img_path}: {e}")

    return processed_image_paths  # return the paths of processed images

def create_labels(base_dir, categories, output_filename, dataset_type):
    """Creates and saves labels for a dataset."""
    labels = []
    image_paths = []  # Keep track of image paths to ensure consistency

    for category in categories:
        category_dir = os.path.join(base_dir, dataset_type, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Directory not found: {category_dir}")
            continue

        # Use glob to find image files with different extensions
        image_files = []
        for ext in ('.png', '.jpg', '.jpeg'):
            image_files.extend(glob.glob(os.path.join(category_dir, f'*{ext}')))

        label_index = categories.index(category)
        labels.extend([label_index] * len(image_files))
        image_paths.extend(image_files)  # Store the full paths

    labels_array = np.array(labels)
    np.save(os.path.join(META_DIR, output_filename), labels_array)
    print(f"Labels saved to {os.path.join(META_DIR, output_filename)}")

    return image_paths  # Return the list of image paths

def load_preprocessed_data(processed_image_paths):
    """Loads preprocessed image data based on the paths of processed images."""
    data = []
    for img_path in processed_image_paths:
        npy_path = os.path.join(PREPROCESSED_DIR, 'Train' if TRAIN_DIR_NAME in img_path else 'Test',
                                os.path.splitext(os.path.basename(img_path))[0] + '.npy')
        try:
            img_data = np.load(npy_path)
            data.append(img_data.flatten())
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue  # Skip to the next file
    return np.array(data)

def apply_pca(data, n_components=None):
    """Applies PCA to reduce dimensionality."""
    if n_components is None:
        n_components = min(data.shape)
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data

def train_random_forest(X_train, y_train, X_test, y_test):
    """Trains and evaluates a Random Forest model with hyperparameter tuning."""

    # Check for consistent number of samples
    if X_train.shape[0] != y_train.shape[0]:
        print("ERROR: Inconsistent number of samples between X_train and y_train.")
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        return  # Exit the function
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_split, y_train_split)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    best_model = grid_search.best_estimator_

    val_predictions = best_model.predict(X_val_split)
    print("Validation Report:\n", classification_report(y_val_split, val_predictions))

    test_predictions = best_model.predict(X_test)
    print("Test Report:\n", classification_report(y_test, test_predictions))

    joblib.dump(best_model, os.path.join(MODEL_DIR, 'fake_currency_rf_model.pkl'))
    print(f"Random Forest model saved to {os.path.join(MODEL_DIR, 'fake_currency_rf_model.pkl')}")

def create_ann_model(input_dim):
    """Creates a simple Artificial Neural Network (ANN) model."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')  # Output layer with softmax for multi-class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Changed loss function
    return model

def train_ann(X_train, y_train, X_test, y_test):
    """Trains and evaluates an Artificial Neural Network (ANN) model."""

    # Check for consistent number of samples before ANN training
    if X_train.shape[0] != y_train.shape[0]:
        print("ERROR: Inconsistent number of samples between X_train and y_train for ANN.")
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        return  # Exit the function

    # Split data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_split, y_train_split)

    # Create the model
    input_dim = X_resampled.shape[1]
    model = create_ann_model(input_dim)
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'best_ann_model.keras'),
                                        monitor='val_accuracy', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(X_resampled, y_resampled,
                        validation_data=(X_val_split, y_val_split),
                        epochs=100, batch_size=32,
                        callbacks=[early_stopping, model_checkpoint], verbose=1)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val_split, y_val_split, verbose=0)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Save the model
    model.save(os.path.join(MODEL_DIR, 'fake_currency_ann_model.keras'))
    print(f"ANN model saved to {os.path.join(MODEL_DIR, 'fake_currency_ann_model.keras')}")

def train_cnn(train_dir, test_dir):
    """Trains a CNN model using image data generators."""

    img_width, img_height = 64, 64
    num_channels = 3

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=(0.7, 1.3),
        channel_shift_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,  # VERY IMPORTANT: Set shuffle to False for test data
        seed=42
    )

    class_labels = list(train_generator.class_indices.keys())
    num_classes = len(class_labels)

    base_model = VGG16(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(img_width, img_height, num_channels)))

    base_model.trainable = False
    fine_tune_at = 12
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_cnn_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    train_steps = int(np.ceil(train_generator.samples / train_generator.batch_size))  # Convert to int
    test_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))    # Convert to int

    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=100,
        validation_data=test_generator,
        validation_steps=test_steps,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Plot training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    test_generator.reset()
    y_pred = model.predict(test_generator, steps=test_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# ------------------------------ Main Function ------------------------------
def main():
    # 1. Create Labels and get Image Paths
    print("Creating labels...")
    train_image_paths = create_labels(BASE_DIR, CATEGORIES, "train_labels.npy", TRAIN_DIR_NAME)
    test_image_paths = create_labels(BASE_DIR, CATEGORIES, "test_labels.npy", TEST_DIR_NAME)

    # 2. Preprocess Data
    print("Preprocessing data...")
    processed_train_image_paths = preprocess_dataset(BASE_DIR, CATEGORIES, TRAIN_DIR_NAME, train_image_paths)
    processed_test_image_paths = preprocess_dataset(BASE_DIR, CATEGORIES, TEST_DIR_NAME, test_image_paths)

    # 3. Load Preprocessed Data
    print("Loading preprocessed data...")
    X_train = load_preprocessed_data(processed_train_image_paths)
    X_test = load_preprocessed_data(processed_test_image_paths)

    # 4. Load Labels
    print("Loading labels...")
    y_train = np.load(os.path.join(META_DIR, "train_labels.npy"))
    y_test = np.load(os.path.join(META_DIR, "test_labels.npy"))

    # 5. Filter labels based on successfully processed images
    y_train = y_train[:len(X_train)]
    y_test = y_test[:len(X_test)]

    # Check lengths before proceeding
    if len(X_train) != len(y_train):
        print(f"ERROR: Mismatch in length: X_train ({len(X_train)}), y_train ({len(y_train)})")
        return
    if len(X_test) != len(y_test):
        print(f"ERROR: Mismatch in length: X_test ({len(X_test)}), y_test ({len(y_test)})")
        return

    # 6. Apply PCA - FIX: Apply PCA only if there are enough samples
    print("Applying PCA...")
    n_components = min(X_train.shape[0], X_train.shape[1], 33)
    if X_train.shape[0] > 1 and X_test.shape[0] > 1:
        pca, X_train_pca = apply_pca(X_train, n_components=n_components)
        X_test_pca = pca.transform(X_test)  # Use the same PCA for the test set
        joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))  # Save PCA model
    else:
        print("Not enough samples to apply PCA.")
        X_train_pca = X_train
        X_test_pca = X_test

    # 7. Train Models
    print("Training models...")
    if X_train_pca is not None and X_test_pca is not None:  # Check if PCA was applied successfully
        train_random_forest(X_train_pca, y_train, X_test_pca, y_test)
        train_ann(X_train_pca, y_train, X_test_pca, y_test)
    else:
        print("Error: Could not load data for training.")

    # CNN training - Requires directory structure
    train_dir = os.path.join(BASE_DIR, TRAIN_DIR_NAME)
    test_dir = os.path.join(BASE_DIR, TEST_DIR_NAME)
    train_cnn(train_dir, test_dir)

if __name__ == "__main__":
    main()
