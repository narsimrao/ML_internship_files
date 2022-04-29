import pickle

class model_accuracy:

    def Model_score(X_test,Y_test,filename):
        loaded_model = pickle.load(open('models/'+filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)
        return result