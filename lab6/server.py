from flask import Flask, request
from SearchEngine import SearchEngine
from QueryResponse import QueryResponse, from_query_result
import json


# --------------------------------|
# In order to execute run:        |
#     `python server.py`          |
# --------------------------------|

app = Flask(__name__)
engine = SearchEngine()


@app.route('/')
def main():
    return json.dumps('Search engine is now running...')


@app.route('/query', methods=['GET'])
def query():
    query_text = request.args['query']

    results = engine.query(query_text)
    results = list(map(from_query_result, results))

    return repr(results)


if __name__ == '__main__':
    app.run(debug=True)
