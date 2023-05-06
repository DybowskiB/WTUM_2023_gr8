from arima_model import prepare_data, train, calculate_mse_plot
import warnings


warnings.filterwarnings("ignore")

# Store_nbr in [1; 54]
# Product family in [0; 32]
print("ARIMA model store sales prediction")
store_nbr = family = -1
while (store_nbr < 1) | (store_nbr > 54) | (family < 0) | (family > 32):
    store_nbr = int(input("Enter store number ([1; 54]): "))
    family = int(input("Enter family product number ([0; 32]): "))
data = prepare_data(store_nbr, family)
result = train(data)
print("Chosen p, d and q values:", result[2], result[3], result[4])
calculate_mse_plot(result[0], result[1], True)
