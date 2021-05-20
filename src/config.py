import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw folder
TRAIN_FEATURES = os.path.join(ROOT_DIR,'input','raw','train_features.csv')
TRAIN_TARGET_SCORED = os.path.join(ROOT_DIR,'input','raw','train_targets_scored.csv')

# processed folder
TRAIN_TRAGET_FOLDS = os.path.join(ROOT_DIR,'input','processed','train_targets_folds.csv')


if __name__ == '__main__':
    print(ROOT_DIR)