import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import textwrap
from errors.classifier_exceptions import ClassifierWithoutCostException, ClassifierWithoutErrorsException
from errors.plot_exceptions import TooManyDimensionsToPlotException
from evaluators.metrics import stratified_k_fold
from utils.decorators import debug


def show_missclassifications(classifier):
    ''' Plots number of missclassifications during data fitting for each epoch.
    
        Parameters:
        ----------
        classifier: Classifier with errors_ property
     '''
    if not hasattr(classifier, 'errors_'):
        raise ClassifierWithoutErrorsException(classifier)
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

def show_sum_of_squared_errors(classifier):
    ''' Plots sum of squared errors during data fitting for each epoch. 
        
        Parameters:
        ----------
        classifier: Classifier with cost_ property
    '''
    if not hasattr(classifier, 'cost_'):
        raise ClassifierWithoutCostException(classifier)
    plt.plot(range(1, len(classifier.cost_) + 1), np.log10(classifier.cost_), marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('log10(Sum of squared errors)')
    plt.show()

def get_matplotlib_2d_decision_regions_plot(X, y, classifier, x_label=None, y_label=None, resolution=0.02):
    ''' Plots decision regions with marked regions for each label.
        Accepts only binary classification.

        Parameters:
        ----------
        X: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

        y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target vectors, where `n_samples` is the number of samples and
                `n_targets` is the number of response variables.

        classifier: Classifier implementing predict function

        x_label: Text of x axis label

        y_label: Text of y axis label

        resolution: Resolution of plotted decision regions
    '''
    # check if data dimension is correct
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.suptitle(f'{repr(classifier)} Decision Regions')
    if X.shape[1] > 2:
        raise TooManyDimensionsToPlotException()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    return plt

@debug
def get_2d_decision_regions_plot(X, y, classifier, title, resolution=0.02, margin=0.25):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    x_range = np.arange(x_min, x_max, resolution)
    y_range = np.arange(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    
    with_proba = hasattr(classifier, 'predict_proba')
    if with_proba:
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    scatters = []
    marker = ['square', 'circle']
    color = ['lightcoral', 'lightsteelblue']
    for idx, cl in enumerate(np.unique(y)):
        scatters.append(go.Scatter(
            x=X[y==cl, 0], y=X[y==cl, 1],
            name=f'Label {cl}',
            mode='markers', marker_symbol=marker[idx], marker_color=color[idx]
        ))

    fig = go.Figure(data=scatters, layout=go.Layout(
        title=go.layout.Title(text=title)
    ))
    fig.update_traces(
        marker_size=12,
        marker_line_width=1.5
    )

    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            showscale=True,
            colorscale='RdBu',
            # opacity=0.4,
            name='Score',
            colorbar=dict(
                title='Label 1 probability' if with_proba else 'Label'
            ),
        )
    )
    fig.data[2].colorbar.x=-0.1
    return fig

@debug
def plot_2d_decision_regions(X, y, classifier, title, resolution=0.02, margin=0.25):
    plt = get_2d_decision_regions_plot(X, y ,classifier, title, resolution, margin)
    plt.show()

@debug
def get_surface(classifier, X):
    max_vals = np.max(X, axis=0)
    min_vals = np.min(X, axis=0)
    x = np.array([min_vals[0], max_vals[0]])
    y = np.array([min_vals[1], max_vals[1]])
    calc_z = lambda x, y: -(classifier.w_[0] + classifier.w_[1] * x + classifier.w_[2] * y) / classifier.w_[3]
    z = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            z[i, j] = calc_z(x[j], y[i])
    return x, y, z

@debug
def get_3d_decision_regions_plot(X, y, classifier, title):
    d = {'x': X[:,0], 'y': X[:,1], 'z': X[:,2], 'label': y, 'predicted_label': classifier.predict(X)}
    df = pd.DataFrame(d)

    def gen_color(val):
        if val == 0:
            return 'red'
        elif val == 2:
            return 'green'
        else:
            return 'blue'

    df['color'] = df.apply(lambda row: gen_color(row['label'] + row['predicted_label']), axis=1)
    df['text'] = df.apply(lambda row: f"Label: {row['label']}; Prediction: {row['predicted_label']}", axis=1)
    
    symbols = {'Label: 1; Prediction: 1': 'square',
                'Label: -1; Prediction: -1': 'circle',
                'Label: 1; Prediction: -1': 'square-open',
                'Label: -1; Prediction: 1': 'circle-open',
    }

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=title)
    ))

    for name in np.unique(df['text']):
        temp_df = df.loc[df['text'] == name]
        fig.add_scatter3d(x=temp_df['x'], y=temp_df['y'], z=temp_df['z'], mode='markers', text=temp_df['text'], marker=dict(size=10, color=temp_df['color'], 
                    symbol=symbols[name],
                    line=dict(color='Black', width=1)
                    ), showlegend=True, name=name)
    return fig

@debug
def get_3d_decision_regions_plot_with_boundary(X, y, classifier, title):
    d = {'x': X[:,0], 'y': X[:,1], 'z': X[:,2], 'label': Y, 'predicted_label': classifier.predict(X)}
    df = pd.DataFrame(d)
    
    def gen_color(val):
        if val == 0:
            return 'red'
        elif val == 2:
            return 'green'
        else:
            return 'blue'

    df['color'] = df.apply(lambda row: gen_color(row['label'] + row['predicted_label']), axis=1)
    df['text'] = df.apply(lambda row: f"Label: {row['label']}; Prediction: {row['predicted_label']}", axis=1)

    symbols = {'Label: 1; Prediction: 1': 'square',
                'Label: -1; Prediction: -1': 'circle',
                'Label: 1; Prediction: -1': 'square-open',
                'Label: -1; Prediction: 1': 'circle-open',
    }

    x, y, z = get_surface(classifier, X)
    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=title)
    ))
    for name in np.unique(df['text']):
        temp_df = df.loc[df['text'] == name]
        fig.add_scatter3d(x=temp_df['x'], y=temp_df['y'], z=temp_df['z'], mode='markers', text=temp_df['text'], marker=dict(size=10, color=temp_df['color'], 
                    symbol=symbols[name],
                    line=dict(color='Black', width=1)
                    ), showlegend=True, name=name)
    fig.add_surface(x=x, y=y, z=z, showscale=False, opacity=0.5)
    return fig

@debug
def plot_3d_decision_regions(X, y, classifier, title):
    fig = get_3d_decision_regions_plot(X, y, classifier, title)
    fig.show()

@debug
def plot_3d_decision_regions_with_boundary(X, y, classifier, title):
    fig = get_3d_decision_regions_plot_with_boundary(X, y, classifier, title)
    fig.show()

@debug
def get_decision_regions_plot(X, y, classifier, title=None, resolution=0.02, with_boundary=False):
    if X.shape[1] == 2:
        return get_2d_decision_regions_plot(X, y, classifier, title, resolution)
    elif X.shape[1] == 3:
        if with_boundary:
            return get_3d_decision_regions_plot_with_boundary(X, y, classifier, title)
        else:
            return get_3d_decision_regions_plot(X, y, classifier, title)
    else:
        raise TooManyDimensionsToPlotException()

@debug
def plot_decision_regions(X, y, classifier, title, resolution=0.02, with_boundary=False):
    fig = get_decision_regions_plot(X, y, classifier, title, resolution, with_boundary)
    fig.show()
    
def plot_stratified_k_fold_means_comparison(classifiers, x_train, y_train, n_splits=10, random_state=None, shuffle=False):
    ''' Plots mean accuracy of stratified K fold of given classificators.
        
        Parameters:
        ----------
        classifiers: List of classifiers implementing fit function
        
        x_train: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

        y_train: Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target vectors, where `n_samples` is the number of samples and
                `n_targets` is the number of response variables.

        n_splits: Number of performed data splits

        random_state: sklearn StratifiedKFold random_state param

        shuffle: sklearn StratifiedKFold shuffle param
    '''
    kfolds = []
    for classifier in classifiers:
        kfold = stratified_k_fold(classifier, x_train, y_train, n_splits, random_state, shuffle)
        kfolds.append(kfold)
    ind = np.arange(len(classifiers))
    fig = plt.figure()
    fig.suptitle('Classifiers comparison: Stratified K Fold')
    ax = fig.add_subplot(111)
    plt.bar(ind, [mean for mean, std in kfolds], width=0.35, label='Mean accuracy')
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels([repr(classifier) for classifier in classifiers])
    plt.show()

def get_matplotlib_plot_histogram(values, xlabels, ylabel, title, text_wrap_length=20):
    ind = np.arange(len(values))
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    plt.bar(ind, values, width=0.35, label=ylabel)
    plt.legend()
    ax.set_xticks(ind)
    labels_fix = lambda x: textwrap.fill(x, text_wrap_length)
    for idx, value in enumerate(values):
        ax.annotate(f'{value:.3f}', xy=(idx-0.05, value+0.01))
    ax.set_xticklabels(map(labels_fix, xlabels))
    return plt

@debug
def get_plot_histogram(values, xlabels, ylabel, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        y=values,
        x=xlabels,
        name='Classifier',
        histfunc='sum'
    ))
    fig.update_layout(
        title_text=title,
        xaxis_title_text='Classifier',
        yaxis_title_text=ylabel,
    )
    return fig

@debug
def plot_histogram(values, xlabels, ylabel, title):
    plt = get_plot_histogram(values, xlabels, ylabel, title)
    plt.show()

@debug
def get_pie_chart(labels, values, title):
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=values
    ))
    fig.update_layout(
        title_text=title,
    )
    return fig

@debug
def plot_pie_chart(labels, values, title):
    fig = get_pie_chart(labels, values, title)
    fig.show()

def plot_matplotlib_confusion_matrix(confusion_matrix, title):
    ''' Plots classifier's confusion matrix.

        Parameters:
        ----------
        confusion_matrix: Generated confusion matrix with structure:
                        [[True False, False Positive][False Negative, True Positive]]
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(title)
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix[i, j],
                    va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('Target label')
    return plt

@debug
def plot_confusion_matrix(confusion_matrix, title):
    fig = ff.create_annotated_heatmap(
        x=[0, 1],
        y=[0, 1],
        z=confusion_matrix,
        showscale=True,
        colorscale='blues',
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        title_text=title,
        xaxis_title_text='Predicted Label',
        yaxis_title_text='Actual Label',
    )
    return fig

def plot_learning_and_validation_accuracy(train_sizes, train_mean, train_std, test_mean, test_std):
    ''' Plots learning and validation accuracy.
        
        Parameters:
        ----------
        train_sizes: array-like of shape (n_ticks,)
                Relative or absolute numbers of training examples that was used to generate the learning curve.
        
        train_mean: Mean of learning curve learning

        train_std: Standard deriviation of learnign curve learning

        test_mean: Mean of learning curve validation

        test_std: Standard deriviation of learnign curve validation
    '''
    plt.plot(train_sizes, train_mean, color='blue', marker='o',
            markersize=5, label='Learning accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                    alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--',
            marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                    alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Learning samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

@debug
def get_roc_curves_plot(classifiers, fprs, tprs, aucs):
    fig = go.Figure()
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for classifier, fpr, tpr, auc in zip(classifiers, fprs, tprs, aucs):
        name = f'{classifier} (AUC={auc:.4f})'
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines', showlegend=True))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
    )
    return fig

@debug
def plot_roc_curves(classifiers, fprs, tprs, aucs):
    fig = get_roc_curves_plot(classifiers, fprs, tprs, aucs)
    fig.show()

@debug
def get_feature_correlation(X, y):
    n_features = X.shape[1]
    feature_columns = [f'X{feature}' for feature in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['label'] = y
    df['label'] = df['label'].astype(str)
    # fig = px.scatter_matrix(df, dimensions=feature_columns, color='label')
    fig = go.Figure()
    fig.add_trace(go.Splom(
                dimensions=[dict(label=col, values=df[col]) for col in feature_columns],
                text=df['label'],
                marker=dict(color=y,
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5)
                ))
    fig.update_layout(
        title='Data correlaction',
    )
    return fig

@debug
def plot_features_correlation(X, y):
    fig = get_feature_correlation(X, y)
    fig.show()

@debug
def get_learning_curve_plot(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, title):
    fig = go.Figure()
    # train fill bottom
    fig.add_trace(go.Scatter(
        x=train_sizes, y=[train_score_mean - train_score_std for train_score_mean, train_score_std in zip(train_scores_mean, train_scores_std)],
        fill=None, 
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0.5, color='rgb(255, 0, 0, 0.1)'),
        mode='lines', line_color='red',
        showlegend=False)
    )
    # train fill top
    fig.add_trace(go.Scatter(
        x=train_sizes, y=[train_score_mean + train_score_std for train_score_mean, train_score_std in zip(train_scores_mean, train_scores_std)],
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0.5, color='rgb(255, 0, 0, 0.1)'),
        mode='lines', line_color='red',
        showlegend=False)
    )
    # train
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean,
                        mode='lines+markers',
                        name='Training Score',
                        marker=dict(
                            color='red'
                        )))
    # test fill bottom
    fig.add_trace(go.Scatter(
        x=train_sizes, y=[test_score_mean - test_score_std for test_score_mean, test_score_std in zip(test_scores_mean, test_scores_std)],
        fill=None, 
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0.5, color='rgb(0, 255, 0, 0.1)'),
        mode='lines', line_color='green',
        showlegend=False)
    )
    # test fill top
    fig.add_trace(go.Scatter(
        x=train_sizes, y=[test_score_mean + test_score_std for test_score_mean, test_score_std in zip(test_scores_mean, test_scores_std)],
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0.5, color='rgb(0, 255, 0, 0.1)'),
        mode='lines', line_color='green',
        showlegend=False)
    )
    # test
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean,
                        mode='lines+markers',
                        name='Test Score',
                        marker=dict(
                            color='green'
                        )))
    fig.update_layout(
        xaxis_title='Train size',
        yaxis_title='Score',
        title=title
    )
    return fig

@debug
def plot_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, title):
    fig = get_learning_curve_plot(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, title)
    fig.show()