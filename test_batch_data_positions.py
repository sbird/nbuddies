# test_batch_data.py
import pickle as pkl

def test_batch_data_positions():
    """Test that positions change across different time steps in batch data."""
    with open("/Users/apiratn1/nbuddies/data/mass_segregation/data_batch0.pkl", "rb") as file:
        data = pkl.load(file)
    
    positions = [data["data"][i][0].position for i in range(len(data["data"]))]
    
    # Print positions for verification
    for i, pos in enumerate(positions):
        print(f"Time step {i}: {pos}")
    
    # Simple assertion: check that not all positions are the same
    assert len(set(map(tuple, positions))) > 1, "All positions are identical across time steps"
# (map(tuple, positions) - Converts each position list to a tuple)
# set(...) - Removes duplicates, keeping only unique positions:
# Checks if there's more than 1 unique position:
if __name__ == "__main__":
    test_batch_data_positions()
    print("Test passed!")