import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

filename = 'WineQT.csv'

df = pd.read_csv(filename)

x = df.drop('quality', axis = 1)    # set features
y = df['quality'] # set labels

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 42) # split into train and test dataset

model = RandomForestRegressor(max_depth = 3, random_state = 42)
model.fit(X_train, y_train) #Train the model

# report training score
train_score = model.score(X_train, y_train) * 100

# report test score
test_score = model.score(X_test, y_test) * 100

# write score to a file 
with open ("metrics.txt", 'w') as outputfile:
    outputfile.write('Training variance explained: %2.1f%%\n' % train_score)
    outputfile.write('Test variance explained: %2.1f%%\n' % test_score)


# Plot feature importances

importances = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ['feature', 'importance'])
feature_df = feature_df.sort_values(by = 'importance', ascending = False)

# image format
sns.set(style = 'whitegrid')

ax = sns.barplot(x = 'importance', y = 'feature', data = feature_df)
ax.set_xlabel('Importances')
ax.set_ylabel('Feature')
ax.set_title('Random Forest Feature Importances')

plt.tight_layout()
plt.savefig('feature_importances.png', dpi = 150)
plt.close()


# Plot Residuals
y_pred = model.predict(X_test) + np.random.normal(0,0.25, len(y_test))
y_jitter = y_test + np.random.normal(0,0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns = ['true', 'pred'])

ax = sns.scatterplot(x = 'true', y = 'pred', data = res_df)
ax.set_aspect('equal')
ax.set_xlabel('True Wine Quality', fontsize = 13)
ax.set_ylabel('Predicted Wine Quality', fontsize = 13)
ax.set_title('Residuals', fontsize = 15)

# make it pretty square aspect ratio

ax.plot([1,10], [1,10], 'black', linewidth = 1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig('residuals.png', dpi = 120)