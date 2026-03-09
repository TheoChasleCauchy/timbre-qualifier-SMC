import pandas as pd
import os
from timbre_mlp import TimbreMLP
from samples_dataset import SamplesDataset
from scipy.stats import pearsonr
from tqdm import tqdm
import yaml

def compute_predictions(embedding_type, hidden_layers_conf, hidden_layer_suffix):

    output_size = 20 # 20 timber traits

    # Model structure
    match embedding_type:
        case "clap_embeddings":
            input_size = 512
        case "clap-music_embeddings":
            input_size = 512
        case "vggish_embeddings":
            input_size = 128
        case "mert_embeddings":
            input_size = 768
    
    model_save_folder = f"./models/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/"
    
    dataset_path = f"data/metadata/RWC/{embedding_type}/{embedding_type}_labels.csv"
    dataset = pd.read_csv(dataset_path)
    timbre_traits_names = dataset.columns[2:].tolist() 
    df = []

    # Compute the predicted values for each sample (row) of the dataset and add it to the dataFrame
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc="Computing predictions for all samples"):
        instrument = row.Instrument
        model_save_path = os.path.join(model_save_folder, f"timbre_model_{embedding_type}_{hidden_layer_suffix}_{instrument.replace(' ', '_')}")

        model = TimbreMLP.load_model(
            f"{model_save_path}/timbre_mlp.pth",
            input_size=input_size,
            hidden_sizes=hidden_layers_conf,
            output_size=output_size,
        )
        
        evalDataset, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

        eval_loss, predicted_values, _ = model.evaluate_model(evalDataloader)

        df.append({
            "Sample": row.Path,
            "Excluded Instrument": instrument,
            **{timber_trait: predicted_values[:,timber_trait_id].item() for timber_trait_id, timber_trait in enumerate(timbre_traits_names)}  # Skip first two columns (Sample and Instrument)
        })
    
    df = pd.DataFrame(df)
    df.to_csv(f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions.csv", index=False)

def compute_errors(embedding_type, hidden_layer_suffix):
    predictions_path = f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions.csv"

    # Read the predictions CSV file
    predictions_df = pd.read_csv(predictions_path)

    # Get the ground truth CSV
    ground_truth_path = f"data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)

    # for each row for each timber_trait column, replace the value by the absolute difference between this value and the ground_truth value
    ground_truth_mapping = {}
    for _, row in ground_truth_df.iterrows():
        ground_truth_mapping[row["RWC Name"]] = row[2:]  # Skip first two columns (Instrument, RWC Name)

    for _, row in predictions_df.iterrows():
        instrument = row["Excluded Instrument"]
        if instrument in ground_truth_mapping:
            ground_truth_values = ground_truth_mapping[instrument]
            normalized_ground_truth_values = (ground_truth_values - 1) / 6.0  # Normalize to [0,1]
            for i, timber_trait in enumerate(ground_truth_df.columns[2:]):
                predictions_df.at[row.name, timber_trait] = abs(row[timber_trait] - normalized_ground_truth_values[i])
        else:
            print(f"Warning: No ground truth found for instrument '{instrument}'")

    # Save the updated predictions with absolute differences
    predictions_df.to_csv(f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions_absolute_errors.csv", index=False)

def get_MAE_per_instrument(embedding_type, hidden_layer_suffix):

    absolute_errors_path = f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions_absolute_errors.csv"

    # Read the predictions CSV file
    absolute_errors_df = pd.read_csv(absolute_errors_path)

    # Compute MAE for each instrument and each timber_trait column
    mae_dict_per_instrument = {}
    for instrument in absolute_errors_df["Excluded Instrument"].unique():
        instrument_df = absolute_errors_df[absolute_errors_df["Excluded Instrument"] == instrument]
        mae_dict_per_instrument[instrument] = {}
        mae_dict_per_instrument[instrument]["Excluded Instrument"] = instrument
        for col in instrument_df.columns[2:]:  # Skip first two columns (Sample and Excluded Instrument)
            mae_dict_per_instrument[instrument][col] = instrument_df[col].mean()

    # Reorder rows by the alphabetical order on the "Excluded Instrument" column
    sorted_instruments = sorted(mae_dict_per_instrument.keys())
    mae_dict_per_instrument = {instrument: mae_dict_per_instrument[instrument] for instrument in sorted_instruments}

    # Add an "Average" column to each instrument's row as the first column
    for instrument in mae_dict_per_instrument:
        mae_dict_per_instrument[instrument]["Average"] = sum([v for k, v in mae_dict_per_instrument[instrument].items() if k != "Excluded Instrument"]) / (len(mae_dict_per_instrument[instrument]) - 1)

    # Add an "Average" row
    average_row = {"Excluded Instrument": "Average"}
    for col in mae_dict_per_instrument[list(mae_dict_per_instrument.keys())[0]].keys():
        if col != "Excluded Instrument":
            average_row[col] = sum([mae_dict_per_instrument[instrument][col] for instrument in mae_dict_per_instrument.keys()]) / len(mae_dict_per_instrument)
    mae_dict_per_instrument["Average"] = average_row

    # Convert to DataFrame and save
    mae_df = pd.DataFrame(mae_dict_per_instrument).T
    # Reorder columns to have Average column first after the Excluded Instrument column
    cols = mae_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index("Average")))
    mae_df = mae_df[cols]

    mae_df.to_csv(f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_maes_per_instrument.csv", index=False)

def compute_correlation(embedding_types, model_hidden_layers):
    print("Computing correlation...")

    corr_dict = {}

    for embedding_type in embedding_types:
        for hidden_layers_conf in model_hidden_layers:
            
            match len(hidden_layers_conf):
                case 0:
                    hidden_layer_suffix = "no_hidden_layers"
                case 1:
                    hidden_layer_suffix = f"single_hidden_layer"
                case _:
                    hidden_layer_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            predictions_path = f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions.csv"

            # Read the predictions CSV file
            predictions_df = pd.read_csv(predictions_path)

            # Get the ground truth CSV
            ground_truth_path = f"resources/metadata/qualities_ground_truth.csv"
            ground_truth_df = pd.read_csv(ground_truth_path)
            timber_traits_names = ground_truth_df.columns[2:]  # Skip first two columns (Instrument, RWC Name)

            ground_truth_mapping = {}
            for _, row in ground_truth_df.iterrows():
                ground_truth_mapping[row["RWC Name"]] = row[timber_traits_names]  # Skip first two columns (Instrument, RWC Name)

            instruments_predictions = predictions_df.groupby("Instrument")
            instrument_mean_preds = {}
            # Get average predictions for each instrument
            for instrument, instrument_df in instruments_predictions:
                instrument_mean_preds[instrument] = instrument_df[timber_traits_names].mean()  # Skip first two columns (Sample and Instrument)

            # Compute correlation for each quality column
            all_predictions = []
            all_ground_truth_values = []
            for instrument, _ in instruments_predictions:
                ground_truth_values = ground_truth_mapping[instrument]
                all_ground_truth_values.extend(ground_truth_values)
                all_predictions.extend(instrument_mean_preds[instrument])

            corr, p_value = pearsonr(all_predictions, all_ground_truth_values)
            match p_value:
                case p_value if p_value < 0.01:
                    corr_str = f"{corr:.3f} **"
                case p_value if p_value < 0.05:
                    corr_str = f"{corr:.3f} *"
                case _:
                    corr_str = f"{corr:.3f}"
            corr_dict[f"{embedding_type}_{hidden_layer_suffix}"] = corr_str

    corr_df = pd.DataFrame(list(corr_dict.items()), columns=["Model", "Correlation"])
    corr_df.to_csv(f"experiments/cross-validation_timbre-model/cross-validation_correlations_all_models.csv", index=False)

def compute_predictions_metrics():

    # Load config.yaml
    with open("experiments/cross-validation_timbre-model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_types = config["embeddings_types"]
    model_hidden_layers = config["model_hidden_layers"]

    for embedding_type in embeddings_types:
        for hidden_layers_conf in model_hidden_layers:
            
            match len(hidden_layers_conf):
                case 0:
                    hidden_layer_suffix = "no_hidden_layers"
                case 1:
                    hidden_layer_suffix = f"single_hidden_layer"
                case _:
                    hidden_layer_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            compute_predictions(embedding_type, hidden_layers_conf, hidden_layer_suffix)
            compute_errors(embedding_type, hidden_layers_conf)
            get_MAE_per_instrument(embedding_type, hidden_layer_suffix)
    
    compute_correlation(embeddings_types, model_hidden_layers)