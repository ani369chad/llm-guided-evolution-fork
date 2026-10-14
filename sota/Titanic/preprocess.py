# Team preprocessing from our Titanic ML/MOGP notebook (Titanic_Final.ipynb),
# so LLM-GE sees exactly the same features as our earlier models.

import pandas as pd


if __name__ == '__main__':
    train_data = pd.read_csv("data/train.csv")

    # Drop text columns and use PassengerId as the index (it is not a feature)
    train_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    train_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

    # Fill missing values with the column mean (Age, Fare) or mode (Embarked)
    train_nan_map = {'Age': train_data['Age'].mean(),
                     'Fare': train_data['Fare'].mean(),
                     'Embarked': train_data['Embarked'].mode()[0]}
    train_data.fillna(value=train_nan_map, inplace=True)

    # Map categorical columns to numbers
    columns_map = {'Embarked': {'C': 0, 'Q': 1, 'S': 2}, 'Sex': {'male': 0, 'female': 1}}
    pd.set_option('future.no_silent_downcasting', True)
    train_data.replace(columns_map, inplace=True)
    train_data = train_data.infer_objects()

    print(train_data.head())

    # index=False so PassengerId is not saved as an extra feature column
    train_data.to_csv('data/processed_train.csv', index=False)
    print(f"Saved data/processed_train.csv with {len(train_data)} rows and columns {list(train_data.columns)}")
