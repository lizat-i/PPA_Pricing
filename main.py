import logging
import sys
import myAnciliaryModule
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np

def main():
    """Main function entry point of the script."""
    args = myAnciliaryModule.parse_arguments()
    myAnciliaryModule.setup_logging(args.verbose)

    logging.info("Starting the script")
    logging.debug(f"Using configuration file: {args.config}")

    # Load configuration
    config = myAnciliaryModule.load_config(args.config)

    # Call the hello world printer function
    myAnciliaryModule.print_hello_world()

    # Main logic here
    try:
        logging.info("Running the main logic...")
        #do_something(config)
        logging.info("load_and_plot_fwc")
        load_and_plot_fwc(config)
        logging.info("load_and_plot_solar_data")
        load_and_plot_per_year_average_ssolar_profile_per_hour(config)
        load_and_plot_hourly_distribution_by_season(config)
        plot_normalized_distribution_by_season(config, 4,18)


    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Script finished successfully")


def load_and_plot_per_year_average_ssolar_profile_per_hour(config: dict):
    """Example placeholder function."""

    logging.info("Entering load_and_plot_solar_data")
    df = myAnciliaryModule.fetch_data_as_dataframe(config['query']['solar'],config['database']['time_series_weather'])
    # Step 1: Convert 'utc_timestamp' to datetime and extract hour and month
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['hour'] = df['utc_timestamp'].dt.hour
    df['month'] = df['utc_timestamp'].dt.month

    # Step 2: Group by hour and month, and calculate average DE_LU_price_day_ahead
    df = df.groupby(['month', 'hour'])['DE_solar_profile'].mean().reset_index()

    # Step 3: Pivot the table for easy plotting (Month as rows, Hour as columns)
    pivot_table = df.pivot(index='month', columns='hour', values='DE_solar_profile')

    # Step 4: Plot the results
    plt.figure(figsize=(12, 8))
    plt.title('Average Price Per Hour for Each Month')
    plt.xlabel('Hour of Day')
    plt.ylabel('DE_solar_profile')

    # Plot each month's average price per hour
    for month in pivot_table.index:
        month_name = calendar.month_abbr[month]
        plt.plot(pivot_table.columns, pivot_table.loc[month], label=f'{month_name}')

    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(24))  # Ensure ticks for each hour of the day
    plt.grid(False)
    plt.tight_layout()
    plt.savefig( f"{config['paths']['output_data']}/{"per_year_average_ssolar_profile_per_hour"}.png")
 
def load_and_plot_hourly_distribution_by_season(config: dict):
    """Example placeholder function."""
    logging.info("Entering load_and_plot_solar_data")
    df = myAnciliaryModule.fetch_data_as_dataframe(config['query']['solar'],config['database']['time_series_weather'])
        # Step 1: Convert 'utc_timestamp' to datetime (if not already done)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['hour'] = df['utc_timestamp'].dt.hour
    df['month'] = df['utc_timestamp'].dt.month

    # Step 2: Define summer and winter seasons
    df['season'] = df['month'].apply(lambda x: 'summer' if 4 <= x <= 8 else 'winter')

    # Step 3: Define hours range (4 to 18)
    hours_range = range(4, 19)  # 19 because the end is exclusive in Python ranges

    # Step 4: Create subplots for each hour in the range (15 subplots)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle('Probability Distribution of DE_solar_profile by Hour and Season (Hours 4-18)', fontsize=16)

    # Step 5: Iterate over each hour in the defined range and plot summer and winter distributions
    for idx, hour in enumerate(hours_range):
        ax = axes[idx // 5, idx % 5]  # Get the appropriate subplot (3 rows x 5 columns)

        #df['DE_solar_profile_log'] = np.log1p(df['DE_solar_profile']) 
        # Filter data for the current hour
        hour_data = df[df['hour'] == hour]

        # Plot for summer and winter on the same subplot
        sns.kdeplot(
            hour_data[hour_data['season'] == 'summer']['DE_solar_profile'],
            ax=ax, label='Summer', color='orange', fill=True, bw_adjust=2
        )
        sns.kdeplot(
            hour_data[hour_data['season'] == 'winter']['DE_solar_profile'],
            ax=ax, label='Winter', color='blue', fill=True, bw_adjust=2
        )

        ax.set_title(f'Hour {hour}')
        ax.set_xlabel('DE_solar_profile')
        ax.set_ylabel('Density')
        ax.set_ylim(0, 5)
        ax.legend(loc='upper right', fontsize=8)

    # Step 6: Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the title
    plt.savefig( f"{config['paths']['output_data']}/{"load_and_plot_per_year_average_ssolar_profile_per_hour"}.png")

def plot_normalized_distribution_by_season(config: dict, start_hour: int = 4, end_hour: int = 18):

    df = myAnciliaryModule.fetch_data_as_dataframe(config['query']['solar'],config['database']['time_series_weather'])
    # Step 1: Convert 'utc_timestamp' to datetime (if not already done)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['hour'] = df['utc_timestamp'].dt.hour
    df['month'] = df['utc_timestamp'].dt.month

    # Step 2: Define summer and winter seasons
    df['season'] = df['month'].apply(lambda x: 'summer' if 4 <= x <= 8 else 'winter')

    # Step 3: Define hours range (based on the selected range)
    hours_range = range(start_hour, end_hour + 1)

    # Step 4: Create subplots for each hour in the range
    n_hours = len(hours_range)
    ncols = 5
    nrows = (n_hours + ncols - 1) // ncols  # Calculate rows needed based on columns and number of hours

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 3), sharex=True, sharey=True)
    fig.suptitle(f'Normalized Discrete Distribution of DE_solar_profile by Hour and Season (Hours {start_hour}-{end_hour})', fontsize=16)

    # Flatten the axes array for easier iteration in case of a single row
    axes = axes.flatten()

    # Step 5: Iterate over each hour in the defined range and plot normalized histograms
    for idx, hour in enumerate(hours_range):
        ax = axes[idx]  # Get the appropriate subplot

        # Filter data for the current hour
        hour_data = df[df['hour'] == hour]

        # Plot normalized histogram for summer and winter on the same subplot
        sns.histplot(
            hour_data[hour_data['season'] == 'summer']['DE_solar_profile'],
            ax=ax, label='Summer', color='orange', bins=20, kde=False, stat='probability', element='step', alpha=0.3
        )
        sns.histplot(
            hour_data[hour_data['season'] == 'winter']['DE_solar_profile'],
            ax=ax, label='Winter', color='blue', bins=20, kde=False, stat='probability', element='step', alpha=0.1
        )

        ax.set_title(f'Hour {hour}')
        ax.set_xlabel('DE_solar_profile')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0,0.8)
        ax.set_ylim(0,0.2)

    # Hide any remaining empty subplots if the total is not perfectly divisible by ncols
    for idx in range(n_hours, len(axes)):
        fig.delaxes(axes[idx])

    # Step 6: Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the title
    plt.savefig( f"{config['paths']['output_data']}/{"plot_normalized_distribution_by_season"}.png")

def load_and_plot_fwc(config: dict):
    """Loads forward curve data, plots prices and volume per month on a single chart, and saves the plot."""
    logging.info("Entering load_and_plot_fwc")    

    # Load data and parse dates
    data_base_ger_fwc = pd.read_csv(config['forward_curves']['base_ger_fwc'], sep=';', decimal=",")
    data_base_ger_fwc['Future'] = pd.to_datetime(data_base_ger_fwc['Future'], format="%d.%m.%Y")

    data_peak_ger_fwc = pd.read_csv(config['forward_curves']['peak_ger_fwc'], sep=';', decimal=",")
    data_peak_ger_fwc['Future'] = pd.to_datetime(data_peak_ger_fwc['Future'], format="%d.%m.%Y")
    data_peak_ger_fwc['Settlement Price'] = pd.to_numeric(data_peak_ger_fwc['Settlement Price'])

    # Prepare data
    data_base_ger_fwc['Month'] = data_base_ger_fwc['Future'].dt.to_period('M')
    data_peak_ger_fwc['Month'] = data_peak_ger_fwc['Future'].dt.to_period('M')

    # Group data by month
    monthly_base = data_base_ger_fwc.groupby('Month').agg({
        'Settlement Price': 'mean',
        'Volume Exchange': 'last'
    }).reset_index()
    monthly_base['Month'] = monthly_base['Month'].dt.to_timestamp()

    monthly_peak = data_peak_ger_fwc.groupby('Month').agg({
        'Settlement Price': 'mean',
        'Volume Exchange': 'last'
    }).reset_index()
    monthly_peak['Month'] = monthly_peak['Month'].dt.to_timestamp()

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot line for Base Settlement Prices
    ax1.plot(monthly_base['Month'], monthly_base['Settlement Price'], marker='o', color='tab:blue', label='Base Settlement Price')

    # Plot line for Peak Settlement Prices
    ax1.plot(monthly_peak['Month'], monthly_peak['Settlement Price'], marker='o', color='tab:green', label='Peak Settlement Price')

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Settlement Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Set x-axis date format
    date_format = DateFormatter('%b-%y')
    ax1.xaxis.set_major_formatter(date_format)

    # Bar chart for Volume on the same axis (base and peak combined)
    ax2 = ax1.twinx()
    ax2.bar(
        monthly_base['Month'], 
        monthly_base['Volume Exchange'], 
        alpha=0.4, 
        color='tab:blue', 
        width=20, 
        label='Base Volume Exchange'
    )
    ax2.bar(
        monthly_peak['Month'], 
        monthly_peak['Volume Exchange'], 
        alpha=0.4, 
        color='tab:green', 
        width=20, 
        label='Peak Volume Exchange'
    )
    ax2.set_ylabel('Volume Exchange', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Title and legends
    fig.suptitle('Monthly Settlement Prices and Volume Exchange')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0,None)


    # Save the figure
    plt.savefig('output/combined_data_fwc.png', dpi=300)

    logging.info("Exiting load_and_plot_fwc")

def monte_carlo_markov_chain_model_solar():
    print("Hello World")

if __name__ == "__main__":
    main()
