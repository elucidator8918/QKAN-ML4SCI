import os
import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def get_files(dataset_path, prefix):
    return [os.path.join(dirname, filename)
            for dirname, _, filenames in os.walk(dataset_path)
            for filename in filenames if filename.startswith(prefix)]

def load_files(file_list, max_files=-1):
    dataframes = []
    files_to_load = file_list if max_files == -1 else file_list[:max_files]
    
    for i, file in enumerate(files_to_load):
        print(f'Adding {file}')
        with h5py.File(file, "r") as f:
            if i == 0:
                print(f"|--> Keys: {list(f.keys())}")
            tmp_df = pd.DataFrame(list(f['jets']), 
                                  columns=[col.decode("utf-8") for col in list(f['jetFeatureNames'])])
            print(f'|--> Number of events/rows: {tmp_df.shape[0]}')
            dataframes.append(tmp_df)
    
    return pd.concat(dataframes, ignore_index=True)

def create_multi_df(df):
    labels = {0: 'j_g', 1: 'j_q', 2: 'j_w', 3: 'j_z', 4: 'j_t', 5: 'j_undef'}
    target = df[list(labels.values())].idxmax(axis=1).map({v: k for k, v in labels.items()})
    features = df.drop(columns=list(labels.values())).reset_index(drop=True)
    features['target'] = target.map(labels)
    return features

def get_dataloaders(dataset_path, train_max=61, val_max=27, batch_size=2048):
    train_files = get_files(dataset_path, 'train')
    val_files = get_files(dataset_path, 'val')

    df_train = load_files(train_files, train_max)
    df_val = load_files(val_files, val_max)

    train_features = create_multi_df(df_train)
    val_features = create_multi_df(df_val)

    features = ['j_ptfrac', 'j_pt', 'j_eta', 'j_mass',
                'j_tau1_b1', 'j_tau2_b1', 'j_tau3_b1', 'j_tau1_b2', 'j_tau2_b2', 'j_tau3_b2', 'j_tau32_b1', 'j_tau32_b2',
                'j_zlogz',
                'j_c1_b0', 'j_c1_b1', 'j_c1_b2', 'j_c2_b1', 'j_c2_b2', 'j_d2_b1', 'j_d2_b2', 'j_d2_a1_b1', 'j_d2_a1_b2',
                'j_m2_b1', 'j_m2_b2', 'j_n2_b1', 'j_n2_b2',
                'j_tau1_b1_mmdt', 'j_tau2_b1_mmdt', 'j_tau3_b1_mmdt', 'j_tau1_b2_mmdt', 'j_tau2_b2_mmdt', 'j_tau3_b2_mmdt',
                'j_tau32_b1_mmdt', 'j_tau32_b2_mmdt',
                'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt',
                'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt',
                'j_mass_trim', 'j_mass_mmdt', 'j_mass_prun', 'j_mass_sdb2', 'j_mass_sdm1',
                'j_multiplicity', 'target']

    train_features = train_features[features]
    val_features = val_features[features]

    scaler = StandardScaler()
    train_input = scaler.fit_transform(train_features.drop(columns='target'))
    test_input = scaler.transform(val_features.drop(columns='target'))

    train_label = pd.get_dummies(train_features['target']).values
    test_label = pd.get_dummies(val_features['target']).values

    train_tensor = torch.tensor(train_input, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_label, dtype=torch.float32)
    test_tensor = torch.tensor(test_input, dtype=torch.float32)
    test_label_tensor = torch.tensor(test_label, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_tensor, test_label_tensor)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader