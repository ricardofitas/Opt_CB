# %%
import csv

import ast
import pandas as pd
import re

# Classification table based on the provided image
classification_table = [
    ("G", (None, 1.8), (None, 0.6)),
    ("F", (1.8, 2.6), (0.6, 1.0)),
    ("E", (2.6, 3.5), (1.0, 1.9)),
    ("D", (3.5, 4.8), (1.9, 2.2)),
    ("B", (4.8, 6.5), (2.2, 3.1)),
    ("C", (6.5, 7.9), (3.1, 4.0)),
    ("A", (7.9, 10.0), (4.0, 5.0)),
    ("K", (10.0, None), (5.0, None))
]


def classify(value, ranges):
    """Classify a value based on given ranges."""
    for label, (low, high) in ranges:
        if (low is None or value >= low) and (high is None or value < high):
            return label
    return "Unknown"

def clean_np_float_strings(s):
    """Replace 'np.float64(x)' or 'numpy.float64(x)' with 'x' in a string."""
    return re.sub(r'(np\.float64|numpy\.float64)\(([^)]+)\)', r'\2', s)


def process_csv(input_file, output_csv, output_xlsx):
    # Read the CSV file
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = []
        
        # Skip the first row
        next(reader, None)
        
        # Define headers
        headers = [
            "Weight/stress_ECT [mm/Pa]", "Weight/stiffness_OoP [mm/Pa]",
            "Thickness_liner_1 [mm]", "Thickness_flute [mm]", "Wavelength [mm]", "Amplitude [mm]", "Thickness_liner_2 [mm]",
            "Weight [mm]", "stress_ECT [Pa]", "stiffness_OoP [Pa]",
            "Inc_bin", "Class_wavelength", "Class_amplitude"
        ]
        
        for row in reader:
            try:
                # Extract and parse the design vector from column index 2
                design_vector_str = clean_np_float_strings(row[2])
                design_vector = ast.literal_eval(design_vector_str)

                
                if isinstance(design_vector, list) and len(design_vector) >= 4:
                    # Extracting the needed values
                    wavelength = float(design_vector[2])  # Third element
                    amplitude = float(design_vector[3])  # Fourth element

                    # Classify each based on the table
                    amplitude_class = classify(amplitude, [(label, amp_range) for label, amp_range, _ in classification_table])
                    wavelength_class = classify(wavelength, [(label, wave_range) for label, _, wave_range in classification_table])

                    # Remove the list format, expand values into separate columns
                    new_row = row[0:2] + design_vector[:5] + row[3:]  # Flatten the design vector
                    new_row.extend([amplitude_class, wavelength_class])  # Add classifications
                    
                    data.append(new_row)
                else:
                    row.extend(["Error", "Error"])  # In case of invalid design vector format
                    data.append(row)
            
            except Exception as e:
                row.extend(["Error", "Error"])  # In case of parsing issues
                data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_csv, index=False)
    df.to_excel(output_xlsx, index=False, sheet_name="Processed Data")
    print(f"Processed CSV saved as {output_csv}")
    print(f"Processed XLSX saved as {output_xlsx}")

# Example usage
roots = ["Prob_prod_fix/EPSO0"]

for rooti in roots:
    input_csv = rooti + "/Iteration_99.csv"  # Replace with your input CSV file
    output_csv = rooti + "/output.csv"  # Desired output file name
    output_xlsx = rooti + "/output.xlsx"  # Desired output file name
    process_csv(input_csv, output_csv, output_xlsx)

