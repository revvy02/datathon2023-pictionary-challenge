# Starter Code for Pictionary Plunger

Sorry for the delay, but here it is!



## Your Model

> Note: If you want to use a different programming language, this is supported. Talk to an organizer like @Akil to get your respective starter code. 

Your model should live in the three functions defined in `solution.py`. 

- The evaluator will call the `new_case` function at the start of a new image.

- The evaluator will call the `guess(x: list[int], y: list[int]) -> str` at every storke. It is your responsibility to return a string that you think the line is. FYI, the exact string is the filename of the test case.

- The `add_score(self, score:int)` function will be called at the end of the each test case with the score. You can use this for easier training, although you should probably do some training yourself.
  
  
  
  ## Local Testing
  
  Although you can obviously test using attorney, you will probably prefer to test locally. Running the `Runner.py` file will launch a local tester where you can type your port number (like in `attorney`). 
  
  The evaluator will use data in the `cases` folder. It's empty in the starter code, so you will need to fill it yourself. Each file needs to an `.ndjson` file with the filename being the category tested. You can see examples in the test cases that you were provided. In fact, you *could* just paste that data directly, although that would cause overfitting. 😅



This challenge has presented some difficulties but we're really excited to see what you come up with. We're here for you so reach out if you need it!
