import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import warnings
warnings.filterwarnings('ignore')

# read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# improper data
r1,c1 = train_data.shape
print('The training data has {} rows and {} columns'.format(r1,c1))
r2,c2 = test_data.shape
print('The validation data has {} rows and {} columns'.format(r2,c2))
print()
print('MISSING VALUES IN TRAINING DATASET:')
print(train_data.isna().sum().nlargest(c1))
print('')
print('MISSING VALUES IN VALIDATION DATASET:')
print(test_data.isna().sum().nlargest(c2))

# data processing
train_data.set_index('PassengerId',inplace=True)
test_data.set_index('PassengerId',inplace=True)

train_data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = train_data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)
test_data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = test_data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

train_data['Age'] =train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] =test_data['Age'].fillna(test_data['Age'].median())

train_data['VIP'] =train_data['VIP'].fillna(False)
test_data['VIP'] =test_data['VIP'].fillna(False)

train_data['HomePlanet'] =train_data['HomePlanet'].fillna('Mars')
test_data['HomePlanet'] =test_data['HomePlanet'].fillna('Mars')

train_data['Destination']=train_data['Destination'].fillna("PSO J318.5-22")
test_data['Destination']=test_data['Destination'].fillna("PSO J318.5-22")

train_data['CryoSleep'] =train_data['CryoSleep'].fillna(False)
test_data['CryoSleep'] =test_data['CryoSleep'].fillna(False)

train_data['Cabin'] =train_data['Cabin'].fillna('T/0/P')
test_data['Cabin'] =test_data['Cabin'].fillna('T/0/P')

# heatmap
plt.figure(figsize=(15,18))
sns.heatmap(train_data.corr(), annot=True)
