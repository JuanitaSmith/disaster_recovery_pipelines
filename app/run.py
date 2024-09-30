import json
import plotly

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import os
import sys

# get current working directory
PROJECT_ROOT = os.getcwd()

# strip file path to the root directory of the project
PROJECT_ROOT = PROJECT_ROOT.split(sep='/src')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/tests')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/notebooks')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/models')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/app')[0]

# add path where scripts are
SOURCE_PATH = os.path.join(
    PROJECT_ROOT, "src"
)

# add paths to the project structure
sys.path.append(PROJECT_ROOT)
sys.path.append(SOURCE_PATH)

from src.utils import *

app = Flask(__name__)

def get_data():
    """ load data model was trained with """

    engine = create_engine(config.path_database)
    conn = engine.connect()
    df = pd.read_sql('select * from messages', con=conn, index_col='id')
    category_names = list(df.select_dtypes(np.int64).columns)
    return df, category_names


def return_graphs(df):
    """ Creates plotly visualizations for training data"""

    # category distribution
    graph_one = []
    df_sum = df.sum(numeric_only=True).sort_values(ascending=True)

    graph_one.append(
        Bar(
            x=df_sum.index,
            y=df_sum,
        )
    )

    layout_one = dict(title=dict(text='Distribution of imbalanced categories',
                                 font=dict(family="Arial",size=14, color="blue")),
                      # title_font=dict(size=14),
                      xaxis=dict(title='Categories to predict'),
                      yaxis=dict(title='Count'),
                      font=dict(size=8),
                      font_family="Courier New",
                      type='category',
                      )

    # genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # genre distribution
    graph_two = []
    graph_two.append(
        Bar(
            x=genre_names,
            y=genre_counts,
        )
    )

    layout_two = dict(title=dict(text='Distribution of GENRE',
                                 font=dict(family="Arial",size=14, color="blue")),
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count'),
                      font=dict(size=10),
                      font_family="Courier New",
                      type='category',
                      )

    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))


    return graphs


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ main page of the website """

    df, categories = get_data()
    graphs = return_graphs(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ Web page showing classification results of a message """

    # load model
    model = joblib.load(config.path_model)
    df, categories = get_data()

    # save user input in query
    query = request.args.get('query')
    genre = request.args.get('genre_input').lower()
    d = {'message': [query], 'genre': [genre]}
    df_query = pd.DataFrame(data=d)

    # use model to predict classification for query
    # classification_labels = model.predict([query])[0]
    classification_labels = model.predict(df_query)[0]
    classification_results = dict(zip(categories, classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        genre=genre,
        classification_result=classification_results
    )


def main():
    """ main routine """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
