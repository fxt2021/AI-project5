import os
import pandas as pd
import matplotlib.pyplot as plt

pic_dir = './pictures'


def plot_metrics(metrics, fusion_level):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(metrics['epoch'], metrics['train_loss'], label='train loss', color='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.plot(metrics['epoch'], metrics['train_acc'], label='train acc')
    ax2.plot(metrics['epoch'], metrics['val_acc'], label='val acc')
    ax2.plot(metrics['epoch'], metrics['picture_val_acc'], label='picture val loss')
    ax2.plot(metrics['epoch'], metrics['text_val_acc'], label='text val loss')
    plt.title(fusion_level.replace('_', ' ') + ' model training metrics')
    plt.legend(loc="lower right")
    pic_file = os.path.join(pic_dir, f'{fusion_level}.png')
    plt.savefig(pic_file)


def plot_compare(feature_fusion_metrics, decision_fusion_metrics):
    plt.figure(figsize=(16, 8))

    def subplot(index, y):
        plt.subplot(2, 2, index)
        plt.plot(feature_fusion_metrics['epoch'], feature_fusion_metrics[y], label='feature fusion')
        plt.plot(decision_fusion_metrics['epoch'], decision_fusion_metrics[y], label='decision fusion')
        plt.xlabel('epoch')
        plt.ylabel(y)
        plt.legend(loc='lower right')

    subplot(1, 'train_acc')
    subplot(2, 'val_acc')
    subplot(3, 'picture_val_acc')
    subplot(4, 'text_val_acc')

    pic_file = os.path.join(pic_dir, 'compare.png')
    plt.savefig(pic_file)


def draw():
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    feature_fusion_metrics_file = './logs/feature_fusion/metrics.csv'
    decision_fusion_metrics_file = './logs/decision_fusion/metrics.csv'
    feature_fusion_metrics = pd.read_csv(feature_fusion_metrics_file)
    decision_fusion_metrics = pd.read_csv(decision_fusion_metrics_file)
    plot_metrics(feature_fusion_metrics, 'feature_fusion')
    plot_metrics(decision_fusion_metrics, 'decision_fusion')
    plot_compare(feature_fusion_metrics, decision_fusion_metrics)
