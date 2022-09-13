import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools



class Metric():
    def __init__(self, n_classes):
        self.targets = np.array([], dtype=np.int)
        self.preds = np.array([], dtype=np.int)
        self.n_classes = n_classes
        
    def append_data(self, targets: np.ndarray, preds: np.ndarray) -> None:
        self.targets = np.concatenate((self.targets, targets))
        self.preds = np.concatenate((self.preds, preds))

    def reset(self):
        self.targets = np.array([], dtype=np.int)
        self.preds = np.array([], dtype=np.int)
    
    def calculate(self):
        eye = np.eye(self.n_classes, dtype=np.int)
        targets = eye[self.targets]
        preds = eye[self.preds]
        uar = recall_score(targets, preds, average=None)
        uf1 = f1_score(targets, preds, average=None)
        war = recall_score(targets, preds, average='weighted')
        return  uf1.mean(), uar.mean(), war

    
def plot_confusion_matrix(targets, pred, target_names=None, normalize=True, labels=True, title='Confusion matrix'):
    cm = confusion_matrix(targets, pred)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.style.use('seaborn')
    cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.savefig(os.path.join(log_dir, 'cm.png'), dpi=300, bbox_inches='tight')
    return fig  

