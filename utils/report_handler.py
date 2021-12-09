from importlib import reload
import dash
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
from dataclasses import dataclass
import pandas as pd
import app.style.style as style
import time


class ReportHandler(object):

    SAMPLE_DATA_PAGE_SIZE=20

    @dataclass
    class ReportElements:
        data_correlation: go.Figure
        usage: list
        learning_curves: list
        accuracy: list
        precision: go.Figure
        recall: go.Figure
        f1: go.Figure
        roc: go.Figure
        log_loss: go.Figure
        confusion_matrixes: list
        decision_regions: list
        sample_data: pd.DataFrame = None

    def __init__(self, report_elements, params):
        self.report_elements = report_elements
        self.params = params
        self.timestamp_id = int(time.time())

    @property
    def sample_data(self):
        return self.report_elements.sample_data

    @sample_data.setter
    def sample_data(self, sample_data):
        self.report_elements.sample_data = sample_data

    @property
    def name(self):
        try:
            return self.sample_data.name
        except Exception:
            return ''

    def get_report_elements(self, app):
        
        elements = dcc.Tabs([
                dcc.Tab(label='Parameters', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_tables()),
                dcc.Tab(label='Data', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                        children=self.get_prepared_data_children()),
                dcc.Tab(label='Usage', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graphs(self.report_elements.usage)),
                dcc.Tab(label='Learning Curves', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graphs(self.report_elements.learning_curves)),
                dcc.Tab(label='Accuracy', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graphs(self.report_elements.accuracy)),
                dcc.Tab(label='Precision', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graph(self.report_elements.precision)),
                dcc.Tab(label='Recall', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graph(self.report_elements.recall)),
                dcc.Tab(label='F1 Score', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graph(self.report_elements.f1)),
                dcc.Tab(label='Roc Curve', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graph(self.report_elements.roc) if not self.report_elements.roc is None else []),
                dcc.Tab(label='Log Loss', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graph(self.report_elements.log_loss)),
                dcc.Tab(label='Confusion Matrixes', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graphs(self.report_elements.confusion_matrixes)),
                dcc.Tab(label='Decision Regions', style=style.TAB_STYLE, selected_style=style.TAB_SELECTED_STYLE,
                    children=self.get_prepared_graphs(self.report_elements.decision_regions)),
            ],
            style={'margin': 20},
            id=f'{self.timestamp_id}_tabs',
        )

        @app.callback(
        Output(f'{self.timestamp_id}_sample_data_datatable', 'data'),
        Input(f'{self.timestamp_id}_sample_data_datatable', "page_current"),
        Input(f'{self.timestamp_id}_sample_data_datatable', "page_size"))
        def update_table(page_current,page_size):
            return self.sample_data.iloc[
                page_current*page_size:(page_current+ 1)*page_size
            ].to_dict('records')

        return elements

    def get_prepared_data_children(self):
        data_children = []
        if not self.report_elements.data_correlation is None:
            data_children.append(self.get_prepared_graph(self.report_elements.data_correlation))
        if not self.report_elements.sample_data is None:
            data_children = data_children + self.get_prepared_data_table()
        return data_children

    def get_prepared_graph(self, fig):
        dcc_graph = dcc.Graph(figure=fig, style={'height': '90vh'})
        return dcc_graph

    def get_prepared_graphs(self, figs):
        dcc_graphs = [dcc.Graph(figure=fig, style={'height': '90vh'}) for fig in figs]
        return dcc_graphs

    def get_prepared_tables(self):
        tables = []
        for classifier, params in self.params.items():
            tables.append(dash.html.Header(classifier, style={
                'textAlign': 'center',
                'marginBottom': 20
            }))
            tables.append(dash_table.DataTable(
                id=classifier,
                columns=[{'name': i, 'id': i} for i in params.columns],
                data=params.to_dict('records'),
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_table={
                    'overflowX': 'auto',
                    'width': 500,
                    'marginLeft': 'auto', 
                    'marginRight': 'auto',
                },
                style_as_list_view=True,
                style_cell={'padding': '5px'},
                style_header={
                    'fontWeight': 'bold'
                },
            ))
            tables.append(dash.html.Div(style={
                'margin': 100,
            }))
        return tables

    def get_prepared_data_table(self):
        elements = [dash.html.Header('Data used to train and test the models', style={
            'textAlign': 'center'
        })]
        df = self.report_elements.sample_data
        elements.append(
            dash_table.DataTable(
                id=f'{self.timestamp_id}_sample_data_datatable',
                columns=[{'name': i, 'id': i} for i in df.columns],
                data=df.to_dict('records'),
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_table={
                    'overflowX': 'auto',
                    'width': 1200,
                    'marginLeft': 'auto', 
                    'marginRight': 'auto',
                },
                style_as_list_view=True,
                style_cell={'padding': '5px'},
                style_header={
                    'fontWeight': 'bold'
                },
                page_current=0,
                page_size=ReportHandler.SAMPLE_DATA_PAGE_SIZE,
                page_action='custom',
            )
        )
        return elements
