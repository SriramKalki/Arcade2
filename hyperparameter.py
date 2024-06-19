from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def build_model(optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=1, verbose=0)

param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'epochs': [10, 20],
    'batch_size': [1, 32]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")