from lstm_model import LSTMModel
from dataset import LSTMDataset
import matplotlib.pyplot as plt
from pickle import load
import pandas as pd
import torch

def main():
    MODEL_PATH = ""
    
    model = LSTMModel()
    trained_traffic_volume_TrafficVolumePrediction = model.load_from_checkpoint(MODEL_PATH)
    trained_traffic_volume_TrafficVolumePrediction.eval()

  
    traffic_volume_test_dataset =  LSTMDataset(test=True)
    traffic_volume_test_dataloader = torch.utils.data.DataLoader(traffic_volume_test_dataset, batch_size=16)
    predicted_result, actual_result = [], []
    for i,j in traffic_volume_test_dataloader:
        print(i.shape,j.shape)
    

    for i, (features,targets) in enumerate(traffic_volume_test_dataloader):
        result = trained_traffic_volume_TrafficVolumePrediction(torch.tensor(features, device="cpu"))
        predicted_result.extend(result.view(-1).tolist())
        actual_result.extend(targets.view(-1).tolist())
        
    scaler = load(open(' .pkl', 'rb'))
    actual_predicted_df = pd.DataFrame(data={"actual":actual_result, "predicted": predicted_result})
    inverse_transformed_values = scaler.inverse_transform(actual_predicted_df)
    actual_predicted_df["actual"] = inverse_transformed_values[:,[0]]
    actual_predicted_df["predicted"] = inverse_transformed_values[:,[1]]
    plt.plot(actual_predicted_df["actual"],'b')
    plt.plot(actual_predicted_df["predicted"],'r')
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()