import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file and return a tuple (evidence, labels).
        -0 Administrative, an integer
        -1 Administrative_Duration, a floating point number
        -2 Informational, an integer
        -3 Informational_Duration, a floating point number
        -4 ProductRelated, an integer
        -5 ProductRelated_Duration, a floating point number
        -6 BounceRates, a floating point number
        -7 ExitRates, a floating point number
        -8 PageValues, a floating point number
        -9 SpecialDay, a floating point number
        -10 Month, an index from 0 (January) to 11 (December)
        -11 OperatingSystems, an integer
        -12 Browser, an integer
        -13 Region, an integer
        -14 TrafficType, an integer
        -15 VisitorType, an integer 0 (not returning) or 1 (returning)
        -16 Weekend, an integer 0 (if false) or 1 (if true)

    """

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        i = 0
        labels = []
        evidence = []

        for row in reader:
            evidence_row = []
            
            if i==20:
                break
            for j in range(6):
                if j%2==0:
                    evidence_row.append(int(row[j]))
                else:
                    evidence_row.append(float(row[j]))
            for j in range(6, 10):
                evidence_row.append(float(row[j]))

            
            # evidence_row.append(10) #month
            if row[10] == "Jan":
                evidence_row.append(0)
            elif row[10] == "Feb":
                evidence_row.append(1)
            elif row[10] == "Mar":
                evidence_row.append(2)
            elif row[10] == "Apr":
                evidence_row.append(3)
            elif row[10] == "May":
                evidence_row.append(4)
            elif row[10] == "June":
                evidence_row.append(5)
            elif row[10] == "Jul":
                evidence_row.append(6)
            elif row[10] == "Aug":
                evidence_row.append(7)
            elif row[10] == "Sep":
                evidence_row.append(8)
            elif row[10] == "Oct":
                evidence_row.append(9)
            elif row[10] == "Nov":
                evidence_row.append(10)
            elif row[10] == "Dec":
                evidence_row.append(11)
            

            for j in range(11, 15):
                evidence_row.append(int(row[j]))

            if row[15] == "New_Visitor":
                evidence_row.append(0)
            else:
                evidence_row.append(1)
            if row[16] == "FALSE":
                evidence_row.append(0)
            else:
                evidence_row.append(1)
            
            if row[17] == "FALSE":
                labels.append(0)
            else:
                labels.append(1)

            evidence.append(evidence_row)
            
        return (evidence, labels)
        


def train_model(evidence, labels):
    """
    Training a KNN Classifer on the data
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):

    positives = 0
    true_positives = 0
    negatives = 0
    true_negatives = 0
    for label, pred in zip(labels, predictions):
        if label == 1:
            positives += 1
            if pred == 1:
                true_positives += 1
        else:
            negatives += 1
            if pred == 0:
                true_negatives += 1

    sensitivity = (true_positives/positives)
    specificity = (true_negatives/negatives)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

