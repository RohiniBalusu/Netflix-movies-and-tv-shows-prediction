# Load the model from the pickle file
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Now you can use the loaded model for prediction
predictions = loaded_model.predict(x)
print(predictions)
