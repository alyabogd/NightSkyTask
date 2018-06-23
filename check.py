import sys
import pandas as pd

from sklearn.metrics import f1_score


def main():
    test_path, predictions_path = sys.argv[1:3]
    test = pd.read_csv(test_path)
    y_true = test['class']

    predictions = pd.read_csv(predictions_path)
    pred_values = predictions['prediction']

    print("macro-averaged f1-score: {}".format(f1_score(y_true, pred_values, average='macro')))


if __name__ == '__main__':
    main()
