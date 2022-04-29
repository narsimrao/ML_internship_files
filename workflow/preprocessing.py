from sklearn.model_selection import train_test_split

class preprocessing:
    def preprocessing(titanic_data):
        titanic_data = titanic_data.drop(columns='Cabin', axis=1)
        titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

        titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
        titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

        X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
        Y = titanic_data['Survived']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        return X_train, X_test, Y_train, Y_test

