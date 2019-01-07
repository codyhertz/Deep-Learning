from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd


feature_names = ['party', 'handicapped-infants', 'water-project-cost-sharing',
                 'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                 'el-salvador-aid', 'religious-groups-in-schools',
                 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                 'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                 'education-spending', 'superfund-right-to-sue', 'crime',
                 'duty-free-exports', 'export-administation-act-south-africa'
                ]

voting_data = pd.read_csv('house-votes-84.data.txt', na_values=['?'], names=feature_names)
voting_data.dropna(inplace=True)
voting_data.replace(('y', 'n'), (1, 0), inplace=True)
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)

all_features = voting_data[feature_names].values
all_classes = voting_data['party'].values

train_features = all_features[:int(len(all_features) * .1 * -1)]
train_labels = all_classes[:int(len(all_features) * .1 * -1)]

test_features = all_features[int(len(all_features) * .1 * -1):]
test_labels = all_classes[int(len(all_features) * .1 * -1):]

model = Sequential()
model.add(Dense(64, input_dim=len(all_features[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_features, train_labels,
                    batch_size=50,
                    epochs=100,
                    verbose=2,
                    validation_data=(test_features, test_labels)
                   )

score = model.evaluate(test_features, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test = model.predict(test_features[0].reshape(1, 17))
if test[0] >= 0.5:
    test = 1
else:
    test = 0

print()
print('1: Democrat\n0: Republican')
print('Politician 1: Predict: {} Real: {}'.format(test, test_labels[0]))
