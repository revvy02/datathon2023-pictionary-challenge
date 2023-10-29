import numpy as np
import json
import os

def align_to_top_left(strokes):
    min_x = min([min(stroke[0]) for stroke in strokes])
    min_y = min([min(stroke[1]) for stroke in strokes])
    return [[[round(x - min_x) for x in stroke[0]], [round(y - min_y) for y in stroke[1]]] for stroke in strokes]

def scale_to_255(strokes):
    max_value = max(max([max(stroke[0]) for stroke in strokes]), max([max(stroke[1]) for stroke in strokes]))
    scale = 255.0 / max_value
    return [[[round(x * scale) for x in stroke[0]], [round(y * scale) for y in stroke[1]]] for stroke in strokes]

def resample_stroke(stroke):
    x, y = np.array(stroke[0]), np.array(stroke[1])
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cum_dist[-1]
    
    if total_dist == 0:
        return [stroke[0], stroke[1]]
    
    num_samples = int(total_dist) + 1  # Adjusted to add more fidelity
    new_points = np.linspace(0, total_dist, num_samples)
    new_x = np.interp(new_points, cum_dist, x)
    new_y = np.interp(new_points, cum_dist, y)
    return [list(map(round, new_x)), list(map(round, new_y))]

def ramer_douglas_peucker(stroke, epsilon):
    dists = []
    for i in range(1, len(stroke[0]) - 1):
        vector_norm = np.linalg.norm([stroke[0][-1] - stroke[0][0], stroke[1][-1] - stroke[1][0]])
        if vector_norm == 0:
            dist = 0
        else:
            dist = np.linalg.norm(np.cross([stroke[0][-1] - stroke[0][0], stroke[1][-1] - stroke[1][0]], 
                                          [stroke[0][0] - stroke[0][i], stroke[1][0] - stroke[1][i]])) / vector_norm
        dists.append(dist)

    if not dists:
        if len(stroke[0]) == 1:  # If the stroke has a single point
            return [stroke[0], stroke[1]]
        return [stroke[0][0::len(stroke[0])-1], stroke[1][0::len(stroke[1])-1]]

    max_dist = max(dists)
    index = dists.index(max_dist) + 1
    if max_dist > epsilon:
        results1 = ramer_douglas_peucker([stroke[0][:index+1], stroke[1][:index+1]], epsilon)
        results2 = ramer_douglas_peucker([stroke[0][index:], stroke[1][index:]], epsilon)
        return [results1[0] + results2[0][1:], results1[1] + results2[1][1:]]
    else:
        return [stroke[0][0::len(stroke[0])-1], stroke[1][0::len(stroke[1])-1]]

def process_drawing(raw_drawing):
    # Extract the strokes from the raw drawing data
    strokes = raw_drawing["strokes"]

    # Apply the preprocessing steps
    strokes = align_to_top_left(strokes)
    strokes = scale_to_255(strokes)
    
    # Resampling after scaling
    strokes = [resample_stroke(stroke) for stroke in strokes]

    # Simplifying each stroke
    strokes = [ramer_douglas_peucker(stroke, epsilon=2.0) for stroke in strokes]

    # Replace the original strokes with the processed strokes
    raw_drawing["drawing"] = strokes

    return raw_drawing

def process_file(input_file, output_file):
    """Process a single file and save the output to another file."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Load the raw drawing data
            raw_drawing = json.loads(line)

            # Process the drawing
            processed_drawing = process_drawing(raw_drawing)

            # Write the processed drawing to the output file
            outfile.write(json.dumps(processed_drawing) + "\n")

def main():
    # Input and output filenames
    # script_dir = os.path.dirname(os.path.realpath(__file__))

    # Input and output folder paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_folder = os.path.join(script_dir, "Archive")
    output_folder = os.path.join(script_dir, "simplified_files")  # Assuming you want to save processed files in 'simplified_files' folder

    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".ndjson"):  # Check file extension
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, "simplified_" + filename)
            process_file(input_file, output_file)

    

if __name__ == "__main__":
    main()
