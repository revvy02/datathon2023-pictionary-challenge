# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

class Solution:
    def __init__(self):
        self.score = 0
        self.strokes = []
        self.model = model

    # this is a signal that a new drawing is about to be sent
    def new_case(self):
        self.score = 0
        pass

    # given a stroke, return a string of your guess
    def guess(self, x: list[int], y: list[int]) -> str:
        return "fortnite"
        
        self.strokes[0].append(x)
        self.strokes[1].append(y)

        output = model([image])

        # find text class

        # return guessed class

        pass

    # this function is called when you get
    def add_score(self, score: int):
        self.score += score
        print(score)
        pass
