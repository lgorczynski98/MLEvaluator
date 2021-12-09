import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import app.style.style as style
import os
import pickle


class Report(object):

    def __init__(self):
        self.app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions=True
        self.app.title = 'ML Evaluator'
        self.elements = {}
        
    def add_page(self, name, elements):
        self.elements[name] = elements

    def prepare_elements(self):
        side_bar = self.prepare_side_bar()
        content = self.prepare_content()
        self.app.layout = html.Div([dcc.Location(id="url"), side_bar, content])

        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            
            elem = pathname[1:]
            if elem in self.elements:
                return self.elements[elem]
            elif elem == '':
                return html.P('This is the main page!')

            # If the user tries to reach a different page, return a 404 message
            return dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ]
            )

    def prepare_side_bar(self):
        side_bar = html.Div(
            [
                dcc.Link(children=[html.H2("ML Evaluator", className="display-4")], href='/', style=style.LINK_TO_MAIN_PAGE_STYLE),
                html.Hr(),
                html.P(
                    "An application to compare performance of different ML algorithms on different datasets", className="lead"
                ),
                dbc.Nav(
                    [dbc.NavLink(data_name, href=f'/{data_name}', active='exact') for data_name in self.elements.keys()],
                    vertical=True,
                    pills=True,
                ),
            ],
            style=style.SIDEBAR_STYLE,
        )
        return side_bar

    def prepare_content(self):
        content = html.Div(id="page-content", style=style.CONTENT_STYLE)
        return content

    def run(self):
        self.app.run_server(debug=True, use_reloader=False, host="0.0.0.0")
        # self.app.run_server(debug=False)
