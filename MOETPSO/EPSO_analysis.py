import os
import numpy as np
import pandas as pd

# Function to find the max values across all CSV files
def find_max_values(base_path, eps_r):
    max_values = [0, 0]  # Initialize max values for each column
    min_values = [np.inf, np.inf]
    for eps in range(eps_r, eps_r + 1):  # For EPSO1 and EPSO2
        for test_number in range(40):  # For each test from 0 to 49
            folder_path = os.path.join(base_path, f"EPSO{eps}", f"EPSO_{test_number}")
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):  # Check if the file is a CSV
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, usecols=[0, 1])
                    max_values = np.maximum(max_values, df.max().to_numpy())
                    min_values = np.minimum(min_values, df.min().to_numpy())

    for eps in range(eps_r, eps_r + 1):  # For EPSO1 and EPSO2
        for test_number in range(1):  # For each test from 0 to 49
            folder_path = os.path.join(base_path, f"GVRP3_{eps}", f"NSGA_{test_number}")
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):  # Check if the file is a CSV
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path, usecols=[0, 1])
                    max_values = np.maximum(max_values, df.max().to_numpy())
                    min_values = np.minimum(min_values, df.min().to_numpy())
    print(max_values.tolist())
    print(min_values.tolist())

    return max_values.tolist(), min_values.tolist()

# Placeholder function for EHVI calculation, to be replaced or refined as needed.
def calculate_ehvi(points, ref_point, iteration):

    points = sorted(points, key=lambda x: x[1], reverse=True)
    ext = [0,0]

    area = np.prod([1-points[0][i] for i in range(2)])
    point_prev = points[0]
    for k, point in enumerate(points[1:]):

        area += np.prod([1 - point[i] - (1 - point_prev[i])*max(0,i) for i in range(2)])
        point_prev = point
        print(k, point, area)
    return area

# Main function to process the files and calculate convergence.
def process_files(base_path, output_csv_path):
    results = []
    for eps in range(1, 3):  # For EPSO1 and EPSO2
        max_values, min_values = find_max_values(base_path, eps)
        max_values = np.log10(max_values)
        min_values = np.log10(min_values)
        print(max_values)
        print(min_values)
        for test_number in range(40):  # For each test from 0 to 49
            conv = 0
            prev_ehvi = None
            for iteration in range(100):  # For each iteration from 0 to 99
                file_path = os.path.join(base_path, f"EPSO{eps}", f"EPSO_{test_number}", f"Iteration_{iteration}.csv")
                if os.path.exists(file_path):

                    # Load the CSV file and select the first two columns.
                    df = pd.read_csv(file_path, usecols=[0, 1])
                    points = df.to_numpy()

                    points = np.log10(points)

                    points = [[(point[j] - min_values[j]) / (max_values[j] - min_values[j]) for j in range(2)] for point in points]

                    ref_point = [1,1]

                    ehvi = calculate_ehvi(points, ref_point, 1)

                    # Check if the previous and current EHVI values are close enough.
                    if prev_ehvi is not None and abs(prev_ehvi - ehvi) < 0.000001:
                        conv += 1
                    else:
                        conv = 0  # Reset conv if the difference is significant.

                    prev_ehvi = ehvi

                    # Store the result for this iteration.
                    results.append((eps, test_number, iteration, ehvi, conv))
                else:
                    # If the file doesn't exist, skip to the next iteration.
                    continue
    # Convert the results into a pandas DataFrame.
    df_results = pd.DataFrame(results, columns=['EPSO', 'Test_Number', 'Iteration', 'EHVI', 'Convergence'])

    # Save the DataFrame to a CSV file.
    df_results.to_csv(base_path + output_csv_path, index=False)

    return df_results

# Main function to process the files and calculate convergence.
def process_files_NSGA(base_path, output_csv_path):
    results = []
    for eps in range(1, 3):  # For EPSO1 and EPSO2
        max_values, min_values = find_max_values(base_path, eps)
        max_values = np.log10(max_values)
        min_values = np.log10(min_values)
        for test_number in range(1):  # For each test from 0 to 49
            conv = 0
            prev_ehvi = None
            for iteration in range(100):  # For each iteration from 0 to 99
                file_path = os.path.join(base_path, f"GVRP3_{eps}", f"NSGA_{test_number}", f"Iteration_{iteration}.csv")
                if os.path.exists(file_path):

                    # Load the CSV file and select the first two columns.
                    df = pd.read_csv(file_path, usecols=[0, 1])
                    points = df.to_numpy()


                    points = np.log10(points)
                    points = [[(point[j] - min_values[j]) / (max_values[j] - min_values[j]) for j in range(2)] for point in points]



                    ref_point = [1,1]
                    print(iteration)
                    ehvi = calculate_ehvi(points, ref_point, iteration)

                    # Check if the previous and current EHVI values are close enough.
                    if prev_ehvi is not None and abs(prev_ehvi - ehvi) < 0.000001:
                        conv += 1
                    else:
                        conv = 0  # Reset conv if the difference is significant.

                    prev_ehvi = ehvi

                    # Store the result for this iteration.
                    results.append((eps, test_number, iteration, ehvi, conv))
                else:
                    # If the file doesn't exist, skip to the next iteration.
                    continue
    # Convert the results into a pandas DataFrame.
    df_results = pd.DataFrame(results, columns=['EPSO', 'Test_Number', 'Iteration', 'EHVI', 'Convergence'])

    # Save the DataFrame to a CSV file.
    df_results.to_csv(base_path + output_csv_path, index=False)

    return df_results


# Assuming the base path is defined as follows (you'll need to adjust this according to your actual path):
base_path = "Final_res"
output_csv_path = "/mnt/data/ehvi_results.csv"  # Adjust this path as needed.
output_csv_path2 = "/mnt/data/ehvi_results2.csv"  # Adjust this path as needed.
output_csv_path3 = "/mnt/data/ehvi_results_"

# Process the files and get the results.
#results = process_files(base_path, output_csv_path)
print("COMPLETE 1")
#results2 = process_files_NSGA(base_path, output_csv_path2)
print("COMPLETE 2")
for al in ['EPSO', 'NSGA-II', 'NBIPOP-aCMAES', 'iCMAES-ILS', 'SHADE', 'LSHADE', 'EBOwithCMAR', 'CJADE', 'HS-ES', 'iLSHADE-RSP', 'MOPSO_CP']:
    results2 = process_files(base_path, output_csv_path3 + al + ".csv")

# Main function to process the files and calculate convergence.
def process_files(base_path, output_csv_path):
    results = []
    for eps in range(1, 3):  # For EPSO1 and EPSO2
        max_values, min_values = find_max_values(base_path, eps)
        max_values = np.log10(max_values)
        min_values = np.log10(min_values)
        for test_number in range(1):  # For each test from 0 to 49
            conv = 0
            prev_ehvi = None
            for iteration in range(100):  # For each iteration from 0 to 99
                file_path = os.path.join(base_path, f"GVRP3_{eps}", f"NSGA_{test_number}", f"Iteration_{iteration}.csv")
                if os.path.exists(file_path):

                    # Load the CSV file and select the first two columns.
                    df = pd.read_csv(file_path, usecols=[0, 1])
                    points = df.to_numpy()


                    points = np.log10(points)
                    points = [[(point[j] - min_values[j]) / (max_values[j] - min_values[j]) for j in range(2)] for point in points]



                    ref_point = [1,1]
                    print(iteration)
                    ehvi = calculate_ehvi(points, ref_point, iteration)

                    # Check if the previous and current EHVI values are close enough.
                    if prev_ehvi is not None and abs(prev_ehvi - ehvi) < 0.000001:
                        conv += 1
                    else:
                        conv = 0  # Reset conv if the difference is significant.

                    prev_ehvi = ehvi

                    # Store the result for this iteration.
                    results.append((eps, test_number, iteration, ehvi, conv))
                else:
                    # If the file doesn't exist, skip to the next iteration.
                    continue
    # Convert the results into a pandas DataFrame.
    df_results = pd.DataFrame(results, columns=['EPSO', 'Test_Number', 'Iteration', 'EHVI', 'Convergence'])

    # Save the DataFrame to a CSV file.
    df_results.to_csv(base_path + output_csv_path, index=False)

    return df_results

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
file_path1 = 'Final_res/mnt/data/ehvi_results.csv'
file_path2 = 'Final_res/mnt/data/ehvi_results2.csv'
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Function to compute average metrics by iteration for a given EPSO value
def average_metrics(df, eps_value, metric):
    filtered_df = df[df['EPSO'] == eps_value]
    avg_metric = filtered_df.groupby('Iteration')[metric].mean().reset_index()
    return avg_metric

# Plotting function
def plot_metric(df1, df2, eps_value, metric):
    avg_metric1 = average_metrics(df1, eps_value, metric)
    avg_metric2 = average_metrics(df2, eps_value, metric)

    plt.figure(figsize=(10, 5))

    # Define the color and linestyle for each dataset
    color = 'black'
    linestyle1 = '-'  # Solid line for Dataset 1
    linestyle2 = '--' # Dashed line for Dataset 2

    if metric == "Convergence":
        plt.plot(avg_metric1['Iteration'], 100 - avg_metric1[metric], label=f'EPSO', color=color, linestyle=linestyle1)
        plt.plot(avg_metric2['Iteration'], 100 - avg_metric2[metric], label=f'NSGA-II', color=color, linestyle=linestyle2)
        plt.ylabel("Convergence Score (%)")
    else:
        plt.plot(avg_metric1['Iteration'], 100*avg_metric1[metric], label=f'EPSO', color=color, linestyle=linestyle1)
        plt.plot(avg_metric2['Iteration'], 100*avg_metric2[metric], label=f'NSGA-II', color=color, linestyle=linestyle2)
        plt.ylabel("Normalized EHVI (%)")
    plt.xlabel('Iteration')
    plt.ylim([0,100])
    plt.title(f'Average {metric} for GVRP #{eps_value}')
    plt.legend()
    plt.show()

# Plot EHVI and Convergence for EPSO = 1 and 2
for eps_value in [1, 2]:
    plot_metric(df1, df2, eps_value, 'EHVI')
    plot_metric(df1, df2, eps_value, 'Convergence')

def is_pareto_efficient(costs):
    """
    Identify the Pareto-efficient points
    :param costs: An (n_points, n_objectives) array
    :return: A boolean array indicating Pareto efficiency for each point.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point that is not dominated by c
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # And keep c itself
    return is_efficient

def plot_combined_pareto_fronts(df1_iter10, df2_iter10, df1_iter100, df2_iter100, eps):
    # Calculate Pareto fronts for all datasets
    pareto_mask_eps10 = is_pareto_efficient(df1_iter10.values)
    pareto_points_eps10 = df1_iter10.values[pareto_mask_eps10]

    pareto_mask_nsga10 = is_pareto_efficient(df2_iter10.values)
    pareto_points_nsga10 = df2_iter10.values[pareto_mask_nsga10]

    pareto_mask_eps100 = is_pareto_efficient(df1_iter100.values)
    pareto_points_eps100 = df1_iter100.values[pareto_mask_eps100]

    pareto_mask_nsga100 = is_pareto_efficient(df2_iter100.values)
    pareto_points_nsga100 = df2_iter100.values[pareto_mask_nsga100]

    # Plotting
    plt.figure(figsize=(8, 6))

    # Pareto fronts for iteration 100
    plt.scatter(pareto_points_eps100[:, 0], pareto_points_eps100[:, 1], color='black', edgecolor='k', label='EPSO Pareto Front', marker='^', s=50)
    plt.scatter(pareto_points_nsga100[:, 0], pareto_points_nsga100[:, 1], facecolors='none', edgecolor='k', label='NSGA-II Pareto Front', marker='^', s=50)

    plt.xlabel(r'$\log(z_1)$')  # Example of using Matplotlib's math rendering
    plt.ylabel(r'$\log(z_2)$')
    gg = 1 if eps < 3 else 2

    plt.title(f'Pareto Fronts - GVRP #{gg}')
    plt.legend()
    plt.show()

# Example usage
def aggregate_points_for_iteration(base_path, eps, iteration):
    points = []  # List to store all points from the given iteration
    n_list = 40 if eps < 3 else 1
    for test_number in range(n_list):  # Assuming 50 tests
        # Constructing folder path based on EPSO and test number
        if eps == 1: folder_path = os.path.join(base_path, f"EPSO1", f"EPSO_{test_number}")
        elif eps == 2: folder_path = os.path.join(base_path, f"EPSO2", f"EPSO_{test_number}")
        elif eps == 3: folder_path = os.path.join(base_path, f"GVRP3_1", f"NSGA_{test_number}")
        elif eps == 4: folder_path = os.path.join(base_path, f"GVRP3_2", f"NSGA_{test_number}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Checking for CSV files
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path, usecols=[0, 1])

                    points.append(np.log10(df.to_numpy()))  # Append the array to the list
                except KeyError:
                    # Handle cases where the expected columns are not found
                    print(f"Columns not found in {file_path}")
                except pd.errors.EmptyDataError:
                    # Handle cases where the CSV is empty or malformed
                    print(f"Empty or malformed data in {file_path}")
    if points:  # Check if there are any points collected
        all_points = np.concatenate(points, axis=0)  # Concatenate all point arrays
        return pd.DataFrame(all_points, columns=['x', 'y'])  # Return as a DataFrame
    else:
        return pd.DataFrame(columns=['x', 'y'])  # Return an empty DataFrame if no points were collected

# Example usage
base_path = "Final_res"
iteration10 = 1
iteration100 = 100

# Aggregate points for iteration 10
#df1_iter10 = aggregate_points_for_iteration(base_path, 1, iteration10)  # For EPSO
#df2_iter10 = aggregate_points_for_iteration(base_path, 3, iteration10)  # For NSGA-II
# Aggregate points for iteration 100
df1_iter200 = aggregate_points_for_iteration(base_path, 1, iteration100)  # For EPSO
df2_iter200 = aggregate_points_for_iteration(base_path, 3, iteration100)  # For NSGA-II
# Aggregate points for iteration 100
df1_iter100 = aggregate_points_for_iteration(base_path, 2, iteration100)  # For EPSO
df2_iter100 = aggregate_points_for_iteration(base_path, 4, iteration100)  # For NSGA-II

# Plot the combined Pareto fronts
plot_combined_pareto_fronts(df1_iter200, df1_iter200, df1_iter200, df2_iter200, 1)
plot_combined_pareto_fronts(df1_iter100, df2_iter100, df1_iter100, df2_iter100, 3)


def aggregate_points_for_iteration_with_values(base_path, eps, iteration):
    points = []  # List to store objective values
    x_all_values = []  # List to store X_all_values

    # Adjust folder path construction based on eps
    n_list = 40 if eps < 3 else 1
    for test_number in range(n_list):  # Assuming 50 tests
        folder_path = os.path.join(base_path, f"EPSO{eps}" if eps in [1, 2] else f"GVRP3_{eps - 2}", f"{'EPSO' if eps in [1, 2] else 'NSGA'}_{test_number}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check for CSV files
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path, usecols=['xline_all_values', 'yline_all_values', 'X_all_values'])
                    # Selecting the specific iteration, if applicable
                    if 'inc' in df.columns:
                        df = df[df['inc'] == iteration]
                    points.append(df[['xline_all_values', 'yline_all_values']].to_numpy())
                    x_all_values.extend(df['X_all_values'].values)
                    print(df['X_all_values'].values)  # Debug print
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if points:  # Check if any points were collected
        all_points = np.concatenate(points, axis=0)
        return pd.DataFrame(all_points, columns=['Objective1', 'Objective2']), np.array(x_all_values)
    else:
        return pd.DataFrame(columns=['Objective1', 'Objective2']), np.array([])

import json

def round_floats_in_string(value_str, precision):
    # Load the string as a list
    values = json.loads(value_str.replace("'", '"'))
    # Round the floating-point numbers to the specified precision
    rounded_values = [round(val, precision) if isinstance(val, float) else val for val in values]
    # Convert back to string representation
    return str(rounded_values)

def save_pareto_points_to_csv(df, values, filename):
    # Apply rounding to 4 decimal places for the DataFrame's numeric columns
    df_rounded = df.round(4)

    # Process X_all_values to round floats within the string representation
    rounded_x_all_values = [round_floats_in_string(val, 8) for val in values]

    # Construct a new DataFrame for saving
    pareto_df = pd.DataFrame(df_rounded, columns=['Objective1', 'Objective2'])
    pareto_df['X_all_values'] = rounded_x_all_values  # Updated to include processed X_all_values

    # Save the DataFrame to CSV
    pareto_df.to_csv(filename, index=False)

#iteration = 100  # Example iteration

# Aggregate points for EPSO and NSGA-II for a specific iteration
#df_eps, x_values_eps = aggregate_points_for_iteration_with_values(base_path, 1, iteration)  # EPSO
#df_nsga, x_values_nsga = aggregate_points_for_iteration_with_values(base_path, 3, iteration)  # NSGA-II

# Calculate Pareto front for EPSO and save
#pareto_mask_eps = is_pareto_efficient(df_eps.values)
#save_pareto_points_to_csv(df_eps[pareto_mask_eps], x_values_eps[pareto_mask_eps], 'pareto_points_eps.csv')

# Calculate Pareto front for NSGA-II and save
#pareto_mask_nsga = is_pareto_efficient(df_nsga.values)
#save_pareto_points_to_csv(df_nsga[pareto_mask_nsga], x_values_nsga[pareto_mask_nsga], 'pareto_points_nsga.csv')
