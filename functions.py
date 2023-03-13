## Write function to train-val-test split filtered DataFrame

def train_val_test_split(df, pqs_per_sign):
    '''
    Takes the train.csv DataFrame containing parquet files grouped by sign and splits each sign into train val & test PARQUETS
    '''

    # Split the data into no_words train, validation, test sets
    train_all = pd.DataFrame()
    val_all = pd.DataFrame()
    test_all = pd.DataFrame()
    for sign in signs:

        # Split the data into train and test sets
        train, test = train_test_split(
            df[df["sign"]==sign][:pqs_per_sign], # values for the train test split
            test_size=0.2, # 20% of data for validation set
            random_state=42 # set random state for reproducibility
        )

        # Split the train set into train and validation sets
        train, val = train_test_split(
            train, # values for train val split
            test_size=0.2, # 30% of train data for validation set
            random_state=42 # set random state for reproducibility
        )
        train_all = pd.concat([train_all, train], ignore_index=True)
        val_all = pd.concat([val_all, val], ignore_index=True)
        test_all = pd.concat([test_all, test], ignore_index=True)
    return train_all, val_all, test_all


def parquets_to_Xy(df, max_frames=100):
    # Loop over rows of the DataFrame and load data
    X_list = []
    y_list = []
    print(len(df))
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i)
        path = row['path']
        sign = row['sign']
        data = load_relevant_data_subset('/kaggle/input/asl-signs/' + path, max_frames=max_frames)
        X_list.append(data)
        y_list.append(sign)

    # Concatenate X_list and y_list into numpy arrays
    print("parquet to rows DONE")
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def load_relevant_data_subset(pq_path, max_frames=100):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path)
    lips = data[(data['type']=='face') & (~data['landmark_index'].isin(lips_landmarks))]
    data = data.drop(lips.index)
    data = data[data_columns]

    n_frames = int(len(data) / ROWS_PER_FRAME)
    # Truncate or zero-pad to max_frames frames
    if n_frames > max_frames:
        data = data.iloc[:max_frames*ROWS_PER_FRAME]
    else:
        # Pad with zeros to max_frames frames
        n_zeros = (max_frames - n_frames) * ROWS_PER_FRAME
        pd_zeros = pd.DataFrame(np.zeros((n_zeros, len(data_columns))), columns=data_columns)
        #print(data)
        #print(pd_zeros)
        data = pd.concat([data, pd_zeros])

    n_frames = int(len(data) / ROWS_PER_FRAME)  # Cast n_frames to integer

    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    # Pad or truncate to max_frames frames
    if n_frames > max_frames:
        data = data[:max_frames]
    else:
        data = np.pad(data, [(0, max_frames-n_frames), (0, 0), (0, 0)], mode='constant')
    # Replace nans with zeros
    data[np.isnan(data)] = 0
    return data.astype(np.float32)
