# Sample participant submission for testing
import json

from flask import Flask, request, jsonify, make_response
from solution import Solution

app = Flask(__name__)
sol = Solution()


# BOILERPLATE
@app.route("/newcase", methods=["POST"])
def new_case():
    sol.new_case()
    return jsonify(success=True)

@app.route("/guess", methods=["POST"])
def guess():
    data = request.get_json()
    data = json.loads(data)

    x, y = tuple(data["stroke"])
    guess = sol.guess(x, y)
    print("guess {}".format(guess))
    ret_data = {"guess": guess}

    response = jsonify(ret_data)
    return response

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    data = json.loads(data)

    score = data["score"]
    print("score {}".format(score))
    sol.add_score(score)
    return jsonify(success=True)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
