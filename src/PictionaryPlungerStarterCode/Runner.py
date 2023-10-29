import json

import requests

from Evaluator import Evaluator


class Runner:
    # test_case_dir should be a string containing the directory of where the test cases are stored
    # for the user, this should be the cases folder in the evaluator's directory
    def __init__(self, test_case_dir="./cases", n_cases=5):
        self.directory = test_case_dir
        self.n = n_cases
        self.total_score = 0
        self.base_url = None
        self.headers = {"Content-Type": "application/json"}
        pass

    def evaluate(self, base_url: str) -> float:
        self.total_score = 0
        self.base_url = base_url
        eval = Evaluator(self.directory)

        # loop n_cases times
        for i in range(self.n):

            # we want to send a new test case, so try sending the message x times
            self.send_new_case_msg()

            # number of strokes in the complete image
            # don't know if we will use this value
            stroke_amt = eval.load_new_case()

            stroke = eval.get_next_stroke()
            while stroke:
                # get guess from model
                guess = self.send_stroke_msg(stroke)
                print("guessed {}".format(guess))

                # score the guess
                score = eval.validate(guess)

                # if the answer is correct, send score to model, add score to total and break this loop
                if score:
                    self.send_score_msg(score)
                    self.total_score += score
                    print("new score {}".format(self.total_score))
                    break

                stroke = eval.get_next_stroke()

            if not stroke:
                self.send_score_msg(0)
                print("new score {}".format(self.total_score))

        # after running all test cases, we need to kill the model and return the total score
        self.kill_graceful()
        return self.total_score
        pass

    # signals to the model that we are starting a new test case
    def send_new_case_msg(self) -> None:
        r = requests.post(self.base_url + "/newcase")

        if not r.ok:
            self.kill_force()
        pass

    # sends the score of the previous test case to the model after a test case is finished
    def send_score_msg(self, score: float) -> None:
        data = {"score": score}
        data = json.dumps(data)

        r = requests.post(self.base_url + "/score", json=data)

        if not r.ok:
            self.kill_force()
        pass

    # sends 1 stroke to the model, then returns the model's guess
    def send_stroke_msg(self, case: list[list[int], list[int]]) -> str:
        data = {"stroke": case}
        data = json.dumps(data)

        r = requests.post(self.base_url + "/guess", json=data)
        if not r.ok:
            self.kill_force()

        d = r.json()
        return d["guess"]
        pass

    # These kill functions are just boilerplate
    def kill_graceful(self):
        # send the server the killmsg
        self.kill(0)

    def kill_force(self):
        # send the server the killmsg
        # this print is for logging/local testing purposes
        print("The runner could not communicate with the server. Terminating...")
        self.kill(-1)

    def kill(self, code: int) -> None:
        if __name__ == "__main__":
            exit(code)


def main():
    port = int(input("Enter your port number: "))
    r = Runner()
    print("Evaluating...")

    base_url = "http://localhost:{}".format(port)
    score = r.evaluate(base_url)
    print("Evaluation complete! Your total score is {}.".format(score))


if __name__ == "__main__":
    main()
