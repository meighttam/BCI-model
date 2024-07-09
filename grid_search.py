from sklearn.preprocessing import Normalizer
import pywt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import code
from tqdm import tqdm
from sklearn.metrics import (
    make_scorer, 
    classification_report, 
    f1_score, 
)


# *** Hyperparameters ***
hyperparams = dict(
    wavelet="db4",
    level=5,
    threshold=1.5,
    norm="l2",
    # 14 different subjects, so 14 folds without shuffle
    n_folds=14, # no touchy!
    C=[0.1, 1, 10, 100],
    gamma=["scale", "auto"],
    kernel=["rbf", "linear", "poly", "sigmoid"],
)


# *** helper functions ***

def select_channels(epochs, labels, threshold):
    """Select channels with significant difference between positive and negative epochs."""
    positive_epochs = epochs[labels == 1]
    negative_epochs = epochs[labels == 0]
    mean_pos_signal = np.mean(positive_epochs, axis=0)
    mean_neg_signal = np.mean(negative_epochs, axis=0)

    stds = []
    for i in range(56):
        std = np.std(mean_pos_signal[i] - mean_neg_signal[i])
        stds.append(std)

    stds = np.array(stds)
    stds[stds < threshold] = 0
    channels = np.where(stds > threshold)[0]
    return channels


def wavelet_transform(data, wavelet, level, threshold):
    """Apply wavelet transform to the data."""
    coeffs = pywt.wavedec(data, wavelet=wavelet, level=level)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return coeffs_thresholded


# *** main function ***

def main():
    # Load the data.
    epochs = np.load("../preproc/train_epochs.npy")
    labels = np.load("../preproc/train_infos.npy")[0]

    # Select channels.
    threshold = hyperparams.pop("threshold")
    channels = select_channels(epochs, labels, threshold)
    epochs = epochs[:, channels]

    # Apply wavelet transform.
    wavelet = hyperparams.pop("wavelet")
    level = hyperparams.pop("level")
    epochs_transformed = []
    for epoch in tqdm(epochs):
        coeffs = wavelet_transform(epoch, wavelet, level, threshold)
        epochs_transformed.append(np.concatenate(coeffs, axis=-1))
    epochs_transformed = np.array(epochs_transformed).reshape(epochs.shape[0], -1)
    # code.interact(local=locals())
    # epochs_transformed = epochs_transformed[:1000, :]
    # labels = labels[:1000]

    # Normalize the data.
    norm = hyperparams.pop("norm")
    if norm is not None:
        normalizer = Normalizer(norm=norm)
        epochs_transformed = normalizer.fit_transform(epochs_transformed)

    # Train the model.
    cv = hyperparams.pop("n_folds")
    kfold = KFold(n_splits=cv, shuffle=False)
    scorer = make_scorer(f1_score)
    scores = GridSearchCV(
        SVC(), 
        hyperparams, 
        cv=kfold, 
        n_jobs=-1, 
        verbose=1,
        scoring=scorer
    )
    scores.fit(epochs_transformed, labels)

    # Print the results.
    print("Best hyperparameters:", scores.best_params_)
    print("Best estimator:", scores.best_estimator_)
    print("Classification report:")
    y_pred = scores.predict(epochs_transformed)
    print(classification_report(labels, y_pred, zero_division=1))


if __name__ == "__main__":
    main()