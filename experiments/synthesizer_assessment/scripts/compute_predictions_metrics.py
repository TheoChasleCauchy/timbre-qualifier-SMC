import pandas as pd
import os
from timbre_mlp import TimbreMLP
from samples_dataset import SamplesDataset
from tqdm import tqdm
import torch
from scipy.linalg import sqrtm
import numpy as np

def compute_predictions_on_TokenSynth(embeddings_type, model_hidden_layers, hidden_layers_suffix):
    embeddings_type = embeddings_type + "_embeddings"
    # Model structure
    match embeddings_type:
        case "clap_embeddings":
            input_size = 512
        case "clap-music_embeddings":
            input_size = 512
        case "vggish_embeddings":
            input_size = 128
        case "mert_embeddings":
            input_size = 768
        case _:
            raise ValueError(f"Unsupported embedding type: {embeddings_type}")
    output_size = 20 # 20 timber traits
    
    model_save_path = f"models/synthesizer_assessment/timbre_model_{embeddings_type}_{hidden_layers_suffix}"
    
    for condition_type in ["text", "audio", "text_audio"]:
        dataset_path = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}/embeddings_labels.csv"
        dataset = pd.read_csv(dataset_path)
        timbre_traits_names = dataset.columns[2:].tolist() 
        df = []

        # Compute the predicted values for each sample (row) of the dataset and add it to the dataFrame
        for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc=f"Computing predictions for all samples for {condition_type}"):
            instrument = row.Instrument

            model = TimbreMLP.load_model(
                f"{model_save_path}/timbre_mlp.pth",
                input_size=input_size,
                hidden_sizes=model_hidden_layers,
                output_size=output_size,
            )
            
            _, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

            _, predicted_values, _ = model.evaluate_model(evalDataloader)

            df.append({
                "Sample": row.Path,
                "Instrument": instrument,
                **{timber_trait: predicted_values[:,timber_trait_id].item() for timber_trait_id, timber_trait in enumerate(timbre_traits_names)}  # Skip first two columns (Sample and Instrument)
            })
        
        df = pd.DataFrame(df)
        os.makedirs(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/", exist_ok=True)
        df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv", index=False)

def compute_predictions_on_RWC(embeddings_type, model_hidden_layers, hidden_layers_suffix):
    embeddings_type = embeddings_type + "_embeddings"
    # Model structure
    match embeddings_type:
        case "clap_embeddings":
            input_size = 512
        case "clap-music_embeddings":
            input_size = 512
        case "vggish_embeddings":
            input_size = 128
        case "mert_embeddings":
            input_size = 768
        case _:
            raise ValueError(f"Unsupported embedding type: {embeddings_type}")
    output_size = 20 # 20 timber traits
    
    model_save_path = f"models/synthesizer_assessment/timbre_model_{embeddings_type}_{hidden_layers_suffix}"
    
    dataset_path = f"data/metadata/RWC/clap_embeddings/{embeddings_type}_labels.csv"
    dataset = pd.read_csv(dataset_path)
    timbre_traits_names = dataset.columns[2:].tolist() 
    df = []

    # Compute the predicted values for each sample (row) of the dataset and add it to the dataFrame
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc=f"Computing predictions for all samples of RWC"):
        instrument = row.Instrument

        model = TimbreMLP.load_model(
            f"{model_save_path}/timbre_mlp.pth",
            input_size=input_size,
            hidden_sizes=model_hidden_layers,
            output_size=output_size,
        )
        
        _, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

        _, predicted_values, _ = model.evaluate_model(evalDataloader)

        df.append({
            "Sample": row.Path,
            "Instrument": instrument,
            **{timber_trait: predicted_values[:,timber_trait_id].item() for timber_trait_id, timber_trait in enumerate(timbre_traits_names)}  # Skip first two columns (Sample and Instrument)
        })
    
    df = pd.DataFrame(df)
    os.makedirs(f"experiments/synthesizer_assessment/results/RWC/", exist_ok=True)
    df.to_csv(f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv", index=False)

def compute_errors():
    for condition_type in ["text", "audio", "text_audio"]:
        predictions_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv"

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
            instrument = row["Instrument"]
            if instrument in ground_truth_mapping:
                ground_truth_values = ground_truth_mapping[instrument]
                normalized_ground_truth_values = (ground_truth_values - 1) / 6.0  # Normalize to [0,1]
                for i, timber_trait in enumerate(ground_truth_df.columns[2:]):
                    predictions_df.at[row.name, timber_trait] = abs(row[timber_trait] - normalized_ground_truth_values[i])
            else:
                print(f"Warning: No ground truth found for instrument '{instrument}'")

        # Save the updated predictions with absolute differences
        predictions_df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_absolute_errors.csv", index=False)

def get_MAE_per_instrument():
    for condition_type in ["text", "audio", "text_audio"]:
        absolute_errors_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_absolute_errors.csv"

        # Read the predictions CSV file
        absolute_errors_df = pd.read_csv(absolute_errors_path)
        instrument_names = absolute_errors_df["Instrument"].unique()

        # Compute MAE for each instrument and each timber_trait column
        mae_dict_per_instrument = {}
        for instrument in instrument_names:
            instrument_df = absolute_errors_df[absolute_errors_df["Instrument"] == instrument]
            mae_dict_per_instrument[instrument] = {}
            mae_dict_per_instrument[instrument]["Instrument"] = instrument
            for col in instrument_df.columns[2:]:  # Skip first two columns (Sample and Instrument)
                mae_dict_per_instrument[instrument][col] = instrument_df[col].mean()

        # Reorder rows by the alphabetical order on the "Instrument" column
        sorted_instruments = sorted(mae_dict_per_instrument.keys())
        mae_dict_per_instrument = {instrument: mae_dict_per_instrument[instrument] for instrument in sorted_instruments}

        # Add an "Average" column to each instrument's row as the first column
        for instrument in mae_dict_per_instrument:
            average = sum([v for k, v in mae_dict_per_instrument[instrument].items() if k != "Instrument"]) / (len(mae_dict_per_instrument[instrument]) - 1)
            mae_dict_per_instrument[instrument]["Average"] = average

        # Add an "Average" row
        average_row = {"Instrument": "Average"}
        for col in mae_dict_per_instrument[list(mae_dict_per_instrument.keys())[0]].keys():
            if col != "Instrument" and col != "Average":
                average_mae = sum([mae_dict_per_instrument[instrument][col] for instrument in mae_dict_per_instrument.keys()]) / len(mae_dict_per_instrument)
                average_std = np.mean(absolute_errors_df[absolute_errors_df["Instrument"].isin(instrument_names)][col].std())
                average_row[col] = f"{average_mae:.3f} ± {average_std:.3f}"

        average_mae = np.mean(absolute_errors_df[absolute_errors_df.columns[2:]].mean())
        average_std = np.mean(absolute_errors_df[absolute_errors_df.columns[2:]].std())
        average_row["Average"] = f"{average_mae:.3f} ± {average_std:.3f}"

        mae_dict_per_instrument["Average"] = average_row

        # Convert to DataFrame and save
        mae_df = pd.DataFrame(mae_dict_per_instrument).T
        # Reorder columns to have Average column first after the Instrument column
        cols = mae_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Average")))
        mae_df = mae_df[cols]

        mae_df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_maes_per_instrument.csv", index=False)

def compute_fad_RWC_Synth(embeddings_type: str):
    embeddings_type = embeddings_type + "_embeddings"

    def load_embeddings(folder_path, instrument_name):
        embeddings = []
        for file in os.listdir(folder_path):
            if instrument_name in file and file.endswith('.pt'):
                emb = torch.load(os.path.join(folder_path, file))
                embeddings.append(emb)
        return torch.stack(embeddings)

    def compute_fad(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = torch.from_numpy(sqrtm(sigma1 @ sigma2))
        if torch.is_complex(covmean):
            covmean = covmean.real
        fad = torch.sum(diff**2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fad

    condition_types = ["text", "audio", "text_audio"]
    fads = {cond: {} for cond in condition_types}
    for condition_type in condition_types:

        metadata_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv"
        metadata_df = pd.read_csv(metadata_path)

        instrument_names = metadata_df["Instrument"].unique()

        for instrument in tqdm(instrument_names, total=len(instrument_names), desc=f"Processing {condition_type}"):

            emb1 = load_embeddings(f"data/RWC/embeddings/{embeddings_type}/", instrument_name=instrument)
            emb2 = load_embeddings(f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}/", instrument_name=instrument)

            mu1, sigma1 = torch.mean(emb1, axis=0), torch.cov(emb1.T)
            mu2, sigma2 = torch.mean(emb2, axis=0), torch.cov(emb2.T)

            fad_score = compute_fad(mu1, sigma1, mu2, sigma2)
            fads[condition_type][instrument] = fad_score.item()

        fads[condition_type]["Mean"] = sum(fads[condition_type].values()) / len(fads[condition_type]) if fads[condition_type] else 0

    # Save fads to a CSV file
    fads_df = pd.DataFrame.from_dict(fads, orient='index')
    # Put the Mean column first after the Condition Type column
    mean_col = fads_df.pop("Mean")
    fads_df.insert(0, "Mean", mean_col)

    fads_df.to_csv("experiments/synthesizer_assessment/results/fad_results.csv", index_label="Condition Type")

def fad_mae_table():
    fad_results = pd.read_csv("experiments/synthesizer_assessment/results/fad_results.csv")
    text_mae = pd.read_csv("experiments/synthesizer_assessment/results/text_conditioned_synthesis/text_maes_per_instrument.csv")
    audio_mae = pd.read_csv("experiments/synthesizer_assessment/results/audio_conditioned_synthesis/audio_maes_per_instrument.csv")
    text_audio_mae = pd.read_csv("experiments/synthesizer_assessment/results/text_audio_conditioned_synthesis/text_audio_maes_per_instrument.csv")

    # Get first two columns of fad table
    fad_mae_table = fad_results[fad_results.columns[:2]]

    # Add a "MAE" column for each condition type
    mae_column = []
    for condition_type in ["text", "audio", "text_audio"]:
        if condition_type == "text":
            # Append with the last value of the column "Average" of the df which corresponds to the average value
            mae_column.append(text_mae["Average"].iloc[-1])
        elif condition_type == "audio":
            mae_column.append(audio_mae["Average"].iloc[-1])
        elif condition_type == "text_audio":
            mae_column.append(text_audio_mae["Average"].iloc[-1])
    fad_mae_table["MAE"] = pd.DataFrame(mae_column)
    fad_mae_table.to_csv("experiments/synthesizer_assessment/results/fad_mae_results.csv", index=False)

def compute_predictions_metrics(embeddings_type: str, model_hidden_layers: list):

    match len(model_hidden_layers):
        case 0:
            hidden_layers_suffix = "no_hidden_layers"
        case 1:
            hidden_layers_suffix = f"single_hidden_layer"
        case _:
            hidden_layers_suffix = f"{len(model_hidden_layers)}_hidden_layers"

    compute_predictions_on_TokenSynth(embeddings_type, model_hidden_layers, hidden_layers_suffix)
    compute_predictions_on_RWC(embeddings_type, model_hidden_layers, hidden_layers_suffix)
    compute_errors()
    get_MAE_per_instrument()
    compute_fad_RWC_Synth(embeddings_type)
    fad_mae_table()