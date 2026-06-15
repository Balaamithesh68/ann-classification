import pickle

one_hot = pickle.load(open('one_hot.pkl', 'rb'))
print('categories:', one_hot.categories_)
print('transform shape:', one_hot.transform([['France']]).shape)
print('feature names:', one_hot.get_feature_names_out(['Geography']))
