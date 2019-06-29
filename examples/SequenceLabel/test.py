from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from sklearn import metrics
y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
labels=[3,3,3,1,2,2,3,1,2,3]
predictions=[3,3,1,2,2,2,3,1,2,3]
p, r, f, sup = metrics.precision_recall_fscore_support(labels, predictions, average='macro')
print(precision_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))