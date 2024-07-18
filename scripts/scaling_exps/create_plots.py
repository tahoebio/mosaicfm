import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

models = ['scgpt-test-9m-full-data', 
            'scgpt-test-9m-0.1data',
            'scgpt-test-9m-0.01data',
            'scgpt-25m-1024-fix-norm-apr24-data', 
            'scgpt-70m-1024-fix-norm-apr24-data',
            'scgpt-70m-0.1xdata',
            'scgpt-70m-0.01xdata',
            'scgpt-70m-0.005xdata',
            'scgpt-70m-1024-fix-norm-bs',
            'scgpt-1_3b-2048-prod'
            ] 

def spearman_flops(csv_path, save_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    global models
    
    model_colors = {
        
        'scgpt-test-9m-full-data':'#0210da', 
            'scgpt-test-9m-0.1data': '#0200ba',
            'scgpt-test-9m-0.01data': '#0230fa',
            'scgpt-25m-1024-fix-norm-apr24-data': '#028ffa', 
            'scgpt-70m-1024-fix-norm-apr24-data':'#91e2ff',
            'scgpt-70m-0.1xdata':'#79b5df',
            'scgpt-70m-0.01xdata':'#11e7ef',
            'scgpt-70m-0.005xdata': '#01c7ef',
            'scgpt-70m-1024-fix-norm-bs': '#add8e6',
            'scgpt-1_3b-2048-prod': '#f59127' }
    
    # curve: Spearman - Flops
    plt.figure(figsize=(10, 6))
    max_points, max_flops, min_spearman = [], [], []


    for column in data.columns:
        if '- _step' in column:
            model = column.replace(' - _step', '')
            spearman_col = f'{model} - metrics/eval/Spearman'
            flops_col = f'{model} - total-flops'
            steps_col = f'{model} - _step'
    

            # Filter out rows with missing values in these columns
            model_data = data[[steps_col, spearman_col, flops_col]].dropna()   

            if '70m' in model or '1_3b' in model:
                #discard early points for big models because they have small spearman  
                model_data = model_data[1:]

            model_data[spearman_col] = 1 - model_data[spearman_col] #for visualization purposes plot 1 - spearman
            
            # Find and plot the point with the maximum Spearman correlation
            min_spearman_idx = model_data[spearman_col].idxmin()
            min_spearman.append(model_data.loc[min_spearman_idx, spearman_col])
            max_flops.append(model_data.loc[min_spearman_idx, flops_col])
            max_points.append((max_flops, min_spearman))

            # Plot
            plt.plot(model_data[flops_col], model_data[spearman_col], color=model_colors[model], marker='o', label=model)

    # plt.plot(max_flops, min_spearman, marker='s', markersize=10, linestyle='dotted', color='black')
    plt.xlabel('#FLOPs')
    plt.ylabel('1 - Spearman')
    plt.title('Spearman vs. FLOPs \n for Different Model Sizes')
    plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_path,'spearman_vs_flops.png'), dpi=300)

    # Show the plot
    plt.show() 


from sklearn.linear_model import LinearRegression
import numpy as np


def spearman_flops_last_points(csv_path, save_path):

    data = pd.read_csv(csv_path)

    global models


    plt.figure(figsize=(10, 6))
    max_points = []
    
    for column in data.columns:
        if '- _step' in column:
            model = column.replace(' - _step', '')        
            spearman_col = f'{model} - metrics/eval/Spearman'
            flops_col = f'{model} - total-flops'
            steps_col = f'{model} - _step'
            
            # Filter out rows with missing values in these columns
            model_data = data[[steps_col, spearman_col, flops_col]].dropna()
            model_data[spearman_col] = 1 - model_data[spearman_col] #for visualization purposes plot 1 - spearman
            
            # Find and plot the point with the maximum Spearman correlation
            min_spearman_idx = model_data[spearman_col].idxmin()
            min_spearman = model_data.loc[min_spearman_idx, spearman_col]
            max_flops = model_data.loc[min_spearman_idx, flops_col]
            max_points.append((max_flops, min_spearman))

            plt.scatter(max_flops, min_spearman, marker='x', lw=10, color='red', zorder=5)
            plt.text(max_flops, min_spearman, f'{model}', fontsize=8, ha='right')


    # Convert max_points to numpy array for regression
    max_points = np.array(max_points)
    X = np.log10(max_points[:, 0].reshape(-1, 1))  # Log scale for FLOPs 
    y = max_points[:, 1]  # Spearman


    # Perform linear regression
    reg = LinearRegression().fit(X, y)

    # Generate points for the regression line
    X_fit = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_fit = reg.predict(X_fit)
    # Plot the regression line
    plt.plot(10**X_fit.flatten(), y_fit, color='blue', linestyle='-', label='Fit Line - all models')

    plt.xlabel('#FLOPs')
    plt.ylabel('1 - Spearman')
    plt.title('Spearman vs. FLOPs \n for Different Model Sizes')
    plt.legend()
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_path, 'spearman_vs_flops_optimal.png'), dpi=300)

    plt.show()




from scipy.optimize import curve_fit
import numpy as np

def spearman_fix_cost(csv_path, save_path):

    data = pd.read_csv(csv_path)

    # Define a function to fit the Spearman vs. FLOPs data
    def fit_func(flops, a, b, c):
        # return a * np.log(flops) + b * flops + c
        return a * np.log(flops)**2 + b * np.log(flops) + c 
    
    # #Flops at which to estimate Spearman values
    flop_points = [5e19, 5e20, 5e21]

    # Store the estimated Spearman values
    estimates = {point: [] for point in flop_points}
    estimates = {}
    model_sizes = {9e6: [],
                25e6:[],
                70e6:[],
                1.3e9: []}
    
    flop_colors = {
        5e19: '#6a07f5',
        5e20: '#1c5c3a',
        5e21: '#13f038' 
    }
    
    # Fit the curve and estimate Spearman values for each model
    for column in data.columns:
        if '- _step' in column:
            model = column.replace(' - _step', '')
            spearman_col = f'{model} - metrics/eval/Spearman'
            flops_col = f'{model} - total-flops'
            steps_col = f'{model} - _step'
            
            estimates[model] = []

            # map model name to its number of params
            if '9m' in model:
                model_sizes[9e6].append(model)
            elif '25m' in model:
                model_sizes[25e6].append(model)
            elif '70m' in model:
                model_sizes[70e6].append(model)
            elif '1_3b' in model:
                model_sizes[1.3e9].append(model)
            
            # Filter out rows with missing values in these columns
            model_data = data[[steps_col, spearman_col, flops_col]].dropna()
            model_data[spearman_col] = 1 - model_data[spearman_col] #for visualization purposes plot 1 - spearman

            if '70m' in model or '1_3b' in model:
                #discard early points for big models because they have small spearman  
                model_data = model_data[2:]
            
            # Fit the curve
            params, _ = curve_fit(fit_func, model_data[flops_col], model_data[spearman_col])

            # Estimate Spearman values at specified FLOPs
            for flop in flop_points:
                estimates[model].append(fit_func(flop, *params))

    # #found manually
    # estimates = {5e+19: [0.43660483179761367, 0.42760039202882183, 0.4442114724348498, 0], 5e+20: [0, 0.42658987764702107, 0.42360088756815273, 0.4351022784633629], 5e+21: [0, 0.42696187359251, 0.4215468927526859, 0.41875591984605354]}
    
    # Fit a second-degree polynomial for each specified FLOP point across different models
    poly_curves = {flop: None for flop in flop_points}

    for i, flop in enumerate(flop_points):
        # In here you should choose which models you want to consider for each #Flops to calculate spearman for. 
        # Because the estimate of spearman for some models for too small/large #Flops could be really faulty. 
        # if i==0:
        #     start, end = 0, 3
        # else:
        #     start, end = 1, 4

        # # model_indices = range(start, end)
        # model_indices = np.log([for model in model_sizes[start:end])
        # spearman_values = estimates[flop][start:end]
        model_indices = np.log(np.array([size for size, models in model_sizes.items() for _ in range(len(models))]))
        spearman_values = [estimates[model][i] for model_arrs in model_sizes.values() for model in model_arrs]


        # Fit a second-degree polynomial
        print(model_indices)
        print(spearman_values)
        coeffs = np.polyfit(model_indices, spearman_values, 2)
        print(coeffs)
        poly_curves[flop] = np.poly1d(coeffs)


    # Plot the estimated Spearman values for each model at specified FLOPs
    plt.figure(figsize=(10, 6))


    for i, flop in enumerate(flop_points):
        if i==0:
            start, end = 0, 3
        if i==1:
            start, end = 1, 4
        if i==2:
            start, end = 2, 4
        model_indices = np.log(np.array([size for size, models in model_sizes.items() for _ in range(len(models))]))
        spearman_values = [estimates[model][i] for model_arrs in model_sizes.values() for model in model_arrs]
        plt.plot(model_indices, spearman_values, 'o', color=flop_colors[flop], label=f'FLOPs = {flop:.0e}')
        print(list(model_sizes.keys())[0])
        model_indices = np.linspace(np.log(list(model_sizes.keys())[0]), np.log(list(model_sizes.keys())[-1]), 100)
        plt.plot(model_indices, poly_curves[flop](model_indices), color=flop_colors[flop])


        # plt.plot(np.log(model_sizes[start:end]), estimates[flop][start:end], 'o', color=flop_colors[flop], label=f'FLOPs = {flop:.0e}')
        # # model_indices = np.linspace(start, end-1, 100)
        # model_indices = np.linspace(np.log(model_sizes[start]), np.log(model_sizes[end-1]), 100)
        # plt.plot(model_indices, poly_curves[flop](model_indices), color=flop_colors[flop])


    # Set labels and title
    plt.xlabel('Model Size')
    plt.ylabel('1 - Spearman')
    plt.title('Estimated Spearman at Different FLOPs')
    plt.legend()
    # plt.xticks(range(len(estimates[flop_points[0]])), [model for model in models], rotation=45)
    plt.xticks(np.log(list(model_sizes.keys())), ['9m', '25m', '70m', '1.3b'])
    plt.tight_layout()


    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_path, 'best_model_at_fixed_cost.png'), dpi=300)
    plt.show()

def parameter_scaling(csv_path, save_path):
    data = pd.read_csv(csv_path)

    global models

    model_params, spearmans = [], []    

    for column in data.columns:
        if '- _step' in column:
            model = column.replace(' - _step', '')

            spearman_col = f'{model} - metrics/eval/Spearman'
            flops_col = f'{model} - total-flops'
            steps_col = f'{model} - _step'
            
            # Filter out rows with missing values in these columns
            model_data = data[[steps_col, spearman_col, flops_col]].dropna()
            
            # Find and plot the point with the maximum Spearman correlation
            spearmans.append(model_data[spearman_col].max())

            if '9m' in model:
                model_params.append(9e6)
            elif '25m' in model:
                model_params.append(25e6)
            elif '70m' in model:
                model_params.append(70e6)
            elif '1_3b' in model:
                model_params.append(1.3e9)


    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    # plt.plot(np.log(np.array(model_sizes)), models_dict.values(), marker='o', label='parameter-scaling')
    plt.plot(np.log(np.array(model_params)), spearmans, marker='o', label='parameter-scaling')
    plt.xlabel('Parameter Scale - ln(#parameters)')
    plt.ylabel('Spearman Correlation')
    plt.title('Performance in terms of Spearman Correlation with respect to number of parameters')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_path, 'params_scale.png'), dpi=300)
    plt.show()

def dataset_scaling(csv_path, save_path):
    # the csv file 
    data = pd.read_csv(csv_path)

    global models

    # perfs
    perfs = {"9M" : {
        43e6: None,
        4.3e6: None,
        530e3: None
        }, 
        "70M-High Diversity" : {
        43e6: None,
        4.3e6: None,
        530e3: None, 
        230e3: None          
        }, 
        "70M-Low Diversity" : {
        230e3: None
        }, 
        }
    
    for column in data.columns:
        if '- _step' in column:
            model = column.replace(' - _step', '')        
            if '9m' in model:
                model_params = '9M'
            elif '70m' in model:
                model_params = '70M-Low Diversity' if 'bs' in model else "70M-High Diversity"
            else:
                print("Model not Supported")
                continue

            spearman_col = f'{model} - metrics/eval/Spearman'
            flops_col = f'{model} - total-flops'
            steps_col = f'{model} - _step'
            
            # Filter out rows with missing values in these columns
            model_data = data[[steps_col, spearman_col, flops_col]].dropna()
            
            # Find and plot the point with the maximum Spearman correlation
            if '0.1x' in model:
                num_cells = 4.3e6
            elif '0.01x' in model:
                num_cells = 530e3
            elif '0.005x' in model or 'bs' in model:
                num_cells = 230e3
            else:
                num_cells = 43e6
            
            perfs[model_params][num_cells] = model_data[spearman_col].max()

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(np.array(list(perfs['9M'].keys()))), perfs['9M'].values(), marker='o', label='9M-Parameter Model')
    plt.plot(np.log(np.array(list(perfs['70M-High Diversity'].keys()))), perfs['70M-High Diversity'].values(), marker='o', label='70M-Parameter-High Diversity Model')
    plt.plot(np.log(np.array(list(perfs['70M-Low Diversity'].keys()))), perfs['70M-Low Diversity'].values(), marker='o', label='70M-Parameter-Low Diversity Model')
    plt.xlabel('Dataset Scale - ln(#samples)')
    plt.ylabel('Spearman Correlation')
    plt.title('Performance in terms of Spearman Correlation with respect to Dataset Scale')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_path, 'dataset_scale.png'), dpi=300)
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting")
    parser.add_argument("--csv_path", required=True, help="path to the preprocessed wandb logs.")
    parser.add_argument("--save_path", required=True,  help="path to save the plots." )
    args = parser.parse_args()

    spearman_flops(args.csv_path, args.save_path)
    spearman_flops_last_points(args.csv_path, args.save_path)
    # spearman_fix_cost(args.csv_path, args.save_path)
    parameter_scaling(args.csv_path, args.save_path)
    dataset_scaling(args.csv_path, args.save_path)