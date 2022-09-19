import time

import numpy as np
import pandas as pd


# Customer Profile Generation
def generate_customer_profiles_table(n_customers, random_state=0):
    np.random.seed(random_state)

    customer_id_properties = []

    # Generate customer properties from random distributions
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        mean_amount = np.random.uniform(5, 100)  # Arbitrary (but sensible) value
        std_amount = mean_amount / 2  # Arbitrary (but sensible) value

        mean_nb_tx_per_day = np.random.uniform(0, 4)  # Arbitrary (but sensible) value

        customer_id_properties.append([customer_id,
                                       x_customer_id, y_customer_id,
                                       mean_amount, std_amount,
                                       mean_nb_tx_per_day])

    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                            'x_customer_id', 'y_customer_id',
                                                                            'mean_amount', 'std_amount',
                                                                            'mean_nb_tx_per_day'])

    return customer_profiles_table


# Terminal Profile Generation
def generate_terminal_profiles_table(n_terminals, random_state=0):
    np.random.seed(random_state)

    terminal_id_properties = []

    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)

        terminal_id_properties.append([terminal_id,
                                       x_terminal_id, y_terminal_id])

    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                            'x_terminal_id', 'y_terminal_id'])

    return terminal_profiles_table


# Association of Customer Profiles to Terminals
def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    # Use numpy arrays in the following to speed up computations

    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)

    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals


# Customer Dataset Generation
def generate_dataset(n_customers=10000, n_terminals=1000000, r=5):
    start_time = time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    print("Time to generate customer profiles table: {0:.2}s".format(time.time() - start_time))

    start_time = time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    print("Time to generate terminal profiles table: {0:.2}s".format(time.time() - start_time))

    start_time = time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals'] = customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time() - start_time))

    return (customer_profiles_table, terminal_profiles_table)


if __name__ == "__main__":
    (customer_profiles_table, terminal_profiles_table) = generate_dataset(n_customers=10000, n_terminals=100000, r=5)
    customer_profiles_table.to_csv("data/customer_profiles.csv.gz", index=False, compression="gzip")
    terminal_profiles_table.to_csv("data/terminal_profiles.csv.gz", index=False, compression="gzip")


    # customer_profiles_table.to_pickle("data/customer_profiles.pkl", protocol=4)  # Save to PKL
    # terminal_profiles_table.to_pickle("data/terminal_profiles.pkl", protocol=4)  # Save to PKL
