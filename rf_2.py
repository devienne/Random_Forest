# RUNNING THE RANDOM FOREST

# create an array containing the labels (or target) data
labels = np.array(sta1_rolling['Disp_rate_ALBH'])

# and drop this column from the data frame
features = sta1_rolling.drop('Disp_rate_ALBH', axis = 1)

# get the name of the features and save it in a list
feature_list = list(features.columns)

# split the data set into traninig set (train_features and test_labels)


# and validation set (test_features and test_labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.50)
rf = RandomForestRegressor(n_estimators = 1)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)

# to get the correct indices from the modeled predictions
axis_x = []
for i in test_features.index.tolist():
    axis_x.append(int(i))


# plot prediction vs. real GPS data
plt.plot(data['alpha'], 'r', label = 'Observation')
plt.plot(axis_x, predictions, 'b.', label = 'Model')
plt.xlabel('Time')
plt.ylabel('GPS displacement rate (mm yr$^{-1}$)')
plt.legend()
plt.savefig('/Users/Jose/Desktop/ALBH.png', dpi = 300)
