import json
import os
import queue
from random import choice, randrange


class Evaluator:
    score = int  # maps client names/IDs to scores
    num_guesses = int
    curr_stroke = -1
    answer = str
    categories = list[str]

    def __init__(self, directory: str):
        self.stroke_q = None
        self.directory = directory
        self.categories = self.get_categories()
        pass

    def load_new_case(self):

        # Choose rando category
        category = choice(self.categories)

        # Within category, choose random case
        case = self.pick_case_from_file(category)
        case = json.loads(case)

        # load strokes from case into the queue
        self.stroke_q = queue.Queue()
        for c in case["strokes"]:
            self.stroke_q.put(list(c))

        self.curr_stroke = 0
        self.answer = category

        # return total number of strokes in image
        return self.stroke_q.qsize()
        pass

    # returns the next stroke in the test case. if there are no more strokes, False is returned.
    def get_next_stroke(self) -> list[list[int], list[int]]:
        if not self.stroke_q.empty():
            self.curr_stroke += 1
            return self.stroke_q.get()
        return False
        pass

    def validate(self, guess: str):
        if guess == self.answer:
            return self.get_score()
        return False
        pass

    def get_score(self) -> float:
        return 100 / (self.curr_stroke ** 2)

    def get_categories(self):
        categories = os.listdir(self.directory)
        return [i.replace(".ndjson", "").strip() for i in categories if ".ndjson" in i]

    # this should take in a category, open the corresponding file, and return a random line from that file
    # idk how this function works. i copied it from stack overflow.
    def pick_case_from_file(self, category: str):
        file = open(self.directory + "/{}.ndjson".format(category))
        line = next(file)
        for num, aline in enumerate(file, 2):
            if randrange(num):
                continue
            line = aline
        return line
