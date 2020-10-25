# Regression---Machine-Learning
Regression - Machine Learning - House Sales Prediction

Regression Algorithm Machine Learning - House Sales in King County USA

This dataset contains house sale prices for King County, which includes Seattle.

It includes homes sold between May 2014 and May 2015.

In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import set_config
set_config(print_changed_only=False)

import warnings
warnings.filterwarnings('ignore')

df =pd.read_csv('kc_house_data.csv')
df.head()
Out[1]:
id	date	price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	...	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long	sqft_living15	sqft_lot15
0	7129300520	20141013T000000	221900.0	3	1.00	1180	5650	1.0	0	0	...	7	1180	0	1955	0	98178	47.5112	-122.257	1340	5650
1	6414100192	20141209T000000	538000.0	3	2.25	2570	7242	2.0	0	0	...	7	2170	400	1951	1991	98125	47.7210	-122.319	1690	7639
2	5631500400	20150225T000000	180000.0	2	1.00	770	10000	1.0	0	0	...	6	770	0	1933	0	98028	47.7379	-122.233	2720	8062
3	2487200875	20141209T000000	604000.0	4	3.00	1960	5000	1.0	0	0	...	7	1050	910	1965	0	98136	47.5208	-122.393	1360	5000
4	1954400510	20150218T000000	510000.0	3	2.00	1680	8080	1.0	0	0	...	8	1680	0	1987	0	98074	47.6168	-122.045	1800	7503
5 rows × 21 columns

DATA EXPLORATION
In [2]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21613 entries, 0 to 21612
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   id             21613 non-null  int64  
 1   date           21613 non-null  object 
 2   price          21613 non-null  float64
 3   bedrooms       21613 non-null  int64  
 4   bathrooms      21613 non-null  float64
 5   sqft_living    21613 non-null  int64  
 6   sqft_lot       21613 non-null  int64  
 7   floors         21613 non-null  float64
 8   waterfront     21613 non-null  int64  
 9   view           21613 non-null  int64  
 10  condition      21613 non-null  int64  
 11  grade          21613 non-null  int64  
 12  sqft_above     21613 non-null  int64  
 13  sqft_basement  21613 non-null  int64  
 14  yr_built       21613 non-null  int64  
 15  yr_renovated   21613 non-null  int64  
 16  zipcode        21613 non-null  int64  
 17  lat            21613 non-null  float64
 18  long           21613 non-null  float64
 19  sqft_living15  21613 non-null  int64  
 20  sqft_lot15     21613 non-null  int64  
dtypes: float64(5), int64(15), object(1)
memory usage: 3.5+ MB
In [3]:
df.isnull().sum()
Out[3]:
id               0
date             0
price            0
bedrooms         0
bathrooms        0
sqft_living      0
sqft_lot         0
floors           0
waterfront       0
view             0
condition        0
grade            0
sqft_above       0
sqft_basement    0
yr_built         0
yr_renovated     0
zipcode          0
lat              0
long             0
sqft_living15    0
sqft_lot15       0
dtype: int64
Dataset is quiet good and looks clean, No Missing Values detected

In [4]:
df.describe().T
Out[4]:
count	mean	std	min	25%	50%	75%	max
id	21613.0	4.580302e+09	2.876566e+09	1.000102e+06	2.123049e+09	3.904930e+09	7.308900e+09	9.900000e+09
price	21613.0	5.400881e+05	3.671272e+05	7.500000e+04	3.219500e+05	4.500000e+05	6.450000e+05	7.700000e+06
bedrooms	21613.0	3.370842e+00	9.300618e-01	0.000000e+00	3.000000e+00	3.000000e+00	4.000000e+00	3.300000e+01
bathrooms	21613.0	2.114757e+00	7.701632e-01	0.000000e+00	1.750000e+00	2.250000e+00	2.500000e+00	8.000000e+00
sqft_living	21613.0	2.079900e+03	9.184409e+02	2.900000e+02	1.427000e+03	1.910000e+03	2.550000e+03	1.354000e+04
sqft_lot	21613.0	1.510697e+04	4.142051e+04	5.200000e+02	5.040000e+03	7.618000e+03	1.068800e+04	1.651359e+06
floors	21613.0	1.494309e+00	5.399889e-01	1.000000e+00	1.000000e+00	1.500000e+00	2.000000e+00	3.500000e+00
waterfront	21613.0	7.541757e-03	8.651720e-02	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	1.000000e+00
view	21613.0	2.343034e-01	7.663176e-01	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	4.000000e+00
condition	21613.0	3.409430e+00	6.507430e-01	1.000000e+00	3.000000e+00	3.000000e+00	4.000000e+00	5.000000e+00
grade	21613.0	7.656873e+00	1.175459e+00	1.000000e+00	7.000000e+00	7.000000e+00	8.000000e+00	1.300000e+01
sqft_above	21613.0	1.788391e+03	8.280910e+02	2.900000e+02	1.190000e+03	1.560000e+03	2.210000e+03	9.410000e+03
sqft_basement	21613.0	2.915090e+02	4.425750e+02	0.000000e+00	0.000000e+00	0.000000e+00	5.600000e+02	4.820000e+03
yr_built	21613.0	1.971005e+03	2.937341e+01	1.900000e+03	1.951000e+03	1.975000e+03	1.997000e+03	2.015000e+03
yr_renovated	21613.0	8.440226e+01	4.016792e+02	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	2.015000e+03
zipcode	21613.0	9.807794e+04	5.350503e+01	9.800100e+04	9.803300e+04	9.806500e+04	9.811800e+04	9.819900e+04
lat	21613.0	4.756005e+01	1.385637e-01	4.715590e+01	4.747100e+01	4.757180e+01	4.767800e+01	4.777760e+01
long	21613.0	-1.222139e+02	1.408283e-01	-1.225190e+02	-1.223280e+02	-1.222300e+02	-1.221250e+02	-1.213150e+02
sqft_living15	21613.0	1.986552e+03	6.853913e+02	3.990000e+02	1.490000e+03	1.840000e+03	2.360000e+03	6.210000e+03
sqft_lot15	21613.0	1.276846e+04	2.730418e+04	6.510000e+02	5.100000e+03	7.620000e+03	1.008300e+04	8.712000e+05
In [5]:
df.describe(include='O')
Out[5]:
date
count	21613
unique	372
top	20140623T000000
freq	142
In [6]:
HouseSalesDesc=[]

for i in df.columns:
    HouseSalesDesc.append([
        i,
        df[i].dtypes,
        df[i].isna().sum(),
        (((df[i].isna().sum())/len(df))*100).round(2),
        df[i].nunique(),
        df[i].drop_duplicates().sample(2).values
    ])
    
pd.DataFrame(data=HouseSalesDesc,columns=[
    'Data Feature', 'Data Types', 'Null','Null Percentages','Unique','Unique Sample'
])
Out[6]:
Data Feature	Data Types	Null	Null Percentages	Unique	Unique Sample
0	id	int64	0	0.0	21436	[7430500301, 1105000588]
1	date	object	0	0.0	372	[20140925T000000, 20150106T000000]
2	price	float64	0	0.0	4028	[377691.0, 649800.0]
3	bedrooms	int64	0	0.0	13	[8, 11]
4	bathrooms	float64	0	0.0	30	[0.0, 4.0]
5	sqft_living	int64	0	0.0	1038	[2050, 3504]
6	sqft_lot	int64	0	0.0	9782	[15624, 36276]
7	floors	float64	0	0.0	6	[1.0, 2.0]
8	waterfront	int64	0	0.0	2	[1, 0]
9	view	int64	0	0.0	5	[1, 4]
10	condition	int64	0	0.0	5	[4, 3]
11	grade	int64	0	0.0	12	[5, 7]
12	sqft_above	int64	0	0.0	946	[1430, 4440]
13	sqft_basement	int64	0	0.0	306	[860, 1620]
14	yr_built	int64	0	0.0	116	[1989, 1990]
15	yr_renovated	int64	0	0.0	70	[1946, 1989]
16	zipcode	int64	0	0.0	70	[98199, 98102]
17	lat	float64	0	0.0	5034	[47.5034, 47.3307]
18	long	float64	0	0.0	752	[-121.925, -122.155]
19	sqft_living15	int64	0	0.0	777	[2300, 2822]
20	sqft_lot15	int64	0	0.0	8689	[8017, 4087]
EXPLORATORY DATA ANALYSIS
In [7]:
#Drop Unessecary columns - Feature Selection

df.drop(columns=['id','date','sqft_living15','sqft_lot15'],inplace=True)
In [8]:
df.head()
Out[8]:
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long
0	221900.0	3	1.00	1180	5650	1.0	0	0	3	7	1180	0	1955	0	98178	47.5112	-122.257
1	538000.0	3	2.25	2570	7242	2.0	0	0	3	7	2170	400	1951	1991	98125	47.7210	-122.319
2	180000.0	2	1.00	770	10000	1.0	0	0	3	6	770	0	1933	0	98028	47.7379	-122.233
3	604000.0	4	3.00	1960	5000	1.0	0	0	5	7	1050	910	1965	0	98136	47.5208	-122.393
4	510000.0	3	2.00	1680	8080	1.0	0	0	3	8	1680	0	1987	0	98074	47.6168	-122.045
In [9]:
((df[['bedrooms','price']].groupby(['bedrooms']).mean())*100).round(2).sort_values(by='bedrooms',ascending=False)
Out[9]:
price
bedrooms	
33	6.400000e+07
11	5.200000e+07
10	8.193333e+07
9	8.939998e+07
8	1.105077e+08
7	9.511847e+07
6	8.255206e+07
5	7.865998e+07
4	6.354195e+07
3	4.662321e+07
2	4.013727e+07
1	3.176429e+07
0	4.095038e+07
In [10]:
((df[['floors','price']].groupby(['floors']).mean())*100).round(2).sort_values(by='floors',ascending=False)
Out[10]:
price
floors	
3.5	9.333125e+07
3.0	5.825260e+07
2.5	1.060346e+08
2.0	6.488912e+07
1.5	5.589806e+07
1.0	4.421806e+07
In [11]:
((df[['grade','price']].groupby(['grade']).mean())*100).round(2).sort_values(by='grade',ascending=False)
Out[11]:
price
grade	
13	3.709615e+08
12	2.191222e+08
11	1.496842e+08
10	1.071771e+08
9	7.735132e+07
8	5.428528e+07
7	4.025903e+07
6	3.019196e+07
5	2.485240e+07
4	2.143810e+07
3	2.056667e+07
1	1.420000e+07
In [12]:
((df[['condition','price']].groupby(['condition']).mean())*100).round(2).sort_values(by='condition',ascending=False)
Out[12]:
price
condition	
5	61241808.94
4	52120039.00
3	54201257.81
2	32728714.53
1	33443166.67
In [13]:
((df[['waterfront','price']].groupby(['waterfront']).mean())*100).round(2).sort_values(by='waterfront',ascending=False)
Out[13]:
price
waterfront	
1	1.661876e+08
0	5.315636e+07
In [14]:
#sns.pairplot(df, hue='price', palette='icefire')
In [15]:
df.corr()
Out[15]:
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long
price	1.000000	0.308350	0.525138	0.702035	0.089661	0.256794	0.266369	0.397293	0.036362	0.667434	0.605567	0.323816	0.054012	0.126434	-0.053203	0.307003	0.021626
bedrooms	0.308350	1.000000	0.515884	0.576671	0.031703	0.175429	-0.006582	0.079532	0.028472	0.356967	0.477600	0.303093	0.154178	0.018841	-0.152668	-0.008931	0.129473
bathrooms	0.525138	0.515884	1.000000	0.754665	0.087740	0.500653	0.063744	0.187737	-0.124982	0.664983	0.685342	0.283770	0.506019	0.050739	-0.203866	0.024573	0.223042
sqft_living	0.702035	0.576671	0.754665	1.000000	0.172826	0.353949	0.103818	0.284611	-0.058753	0.762704	0.876597	0.435043	0.318049	0.055363	-0.199430	0.052529	0.240223
sqft_lot	0.089661	0.031703	0.087740	0.172826	1.000000	-0.005201	0.021604	0.074710	-0.008958	0.113621	0.183512	0.015286	0.053080	0.007644	-0.129574	-0.085683	0.229521
floors	0.256794	0.175429	0.500653	0.353949	-0.005201	1.000000	0.023698	0.029444	-0.263768	0.458183	0.523885	-0.245705	0.489319	0.006338	-0.059121	0.049614	0.125419
waterfront	0.266369	-0.006582	0.063744	0.103818	0.021604	0.023698	1.000000	0.401857	0.016653	0.082775	0.072075	0.080588	-0.026161	0.092885	0.030285	-0.014274	-0.041910
view	0.397293	0.079532	0.187737	0.284611	0.074710	0.029444	0.401857	1.000000	0.045990	0.251321	0.167649	0.276947	-0.053440	0.103917	0.084827	0.006157	-0.078400
condition	0.036362	0.028472	-0.124982	-0.058753	-0.008958	-0.263768	0.016653	0.045990	1.000000	-0.144674	-0.158214	0.174105	-0.361417	-0.060618	0.003026	-0.014941	-0.106500
grade	0.667434	0.356967	0.664983	0.762704	0.113621	0.458183	0.082775	0.251321	-0.144674	1.000000	0.755923	0.168392	0.446963	0.014414	-0.184862	0.114084	0.198372
sqft_above	0.605567	0.477600	0.685342	0.876597	0.183512	0.523885	0.072075	0.167649	-0.158214	0.755923	1.000000	-0.051943	0.423898	0.023285	-0.261190	-0.000816	0.343803
sqft_basement	0.323816	0.303093	0.283770	0.435043	0.015286	-0.245705	0.080588	0.276947	0.174105	0.168392	-0.051943	1.000000	-0.133124	0.071323	0.074845	0.110538	-0.144765
yr_built	0.054012	0.154178	0.506019	0.318049	0.053080	0.489319	-0.026161	-0.053440	-0.361417	0.446963	0.423898	-0.133124	1.000000	-0.224874	-0.346869	-0.148122	0.409356
yr_renovated	0.126434	0.018841	0.050739	0.055363	0.007644	0.006338	0.092885	0.103917	-0.060618	0.014414	0.023285	0.071323	-0.224874	1.000000	0.064357	0.029398	-0.068372
zipcode	-0.053203	-0.152668	-0.203866	-0.199430	-0.129574	-0.059121	0.030285	0.084827	0.003026	-0.184862	-0.261190	0.074845	-0.346869	0.064357	1.000000	0.267048	-0.564072
lat	0.307003	-0.008931	0.024573	0.052529	-0.085683	0.049614	-0.014274	0.006157	-0.014941	0.114084	-0.000816	0.110538	-0.148122	0.029398	0.267048	1.000000	-0.135512
long	0.021626	0.129473	0.223042	0.240223	0.229521	0.125419	-0.041910	-0.078400	-0.106500	0.198372	0.343803	-0.144765	0.409356	-0.068372	-0.564072	-0.135512	1.000000
SPLITTING DATA
In [16]:
from sklearn.model_selection import train_test_split

# Algorithm Model

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
In [17]:
X = df.drop(columns='price')  ### Features 
y = df['price'] ### Target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .80, random_state = 42)
Machine Learning Modelling
Linear Regression
In [18]:
from sklearn.linear_model import LinearRegression
In [19]:
model_linreg = LinearRegression()
In [20]:
model_linreg.fit(X_train, y_train)
Out[20]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
In [21]:
model_linreg.predict(X_test)
Out[21]:
array([ 461291.61257276,  731704.09137437, 1217116.74900676, ...,
        315521.37054524,  462833.47300744,  692663.01833351])
In [22]:
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
In [23]:
y_model_Linreg = model_linreg.predict(X_test)
In [24]:
r2_model_Linreg= (r2_score(y_test, y_model_Linreg)).round(2)
MAE_model_linreg = mean_absolute_error(y_test, y_model_Linreg)
MSE_model_linreg= mean_squared_error(y_test, y_model_Linreg)
RMSE_model_linreg = np.sqrt(MSE_model_linreg)

print("Evaluation Matrix Linear regression")
print("MAE Score: ", MAE_model_linreg)
print("MSE Score: ", MSE_model_linreg)
print("RMSE Score: ", RMSE_model_linreg)
print("R2 : ", r2_model_Linreg)
Evaluation Matrix Linear regression
MAE Score:  123562.25603768413
MSE Score:  42821101853.71838
RMSE Score:  206932.60220109925
R2 :  0.69
K Nearest Neighbors
In [25]:
from sklearn.neighbors import KNeighborsRegressor
In [26]:
model_KNN = KNeighborsRegressor()
In [27]:
model_KNN.fit(X_train, y_train)
Out[27]:
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                    weights='uniform')
In [28]:
y_model_KNN = model_KNN.predict(X_test)
In [29]:
r2_model_KNN= (r2_score(y_test, y_model_KNN)).round(2)
MAE_model_KNN = mean_absolute_error(y_test, y_model_KNN)
MSE_model_KNN= mean_squared_error(y_test, y_model_KNN)
RMSE_model_KNN= np.sqrt(MSE_model_KNN)

print("Evaluation Matrix K Nearest Neighbors")
print("MAE Score: ", MAE_model_KNN)
print("MSE Score: ", MSE_model_KNN)
print("RMSE Score: ", RMSE_model_KNN)
print("R2 : ", r2_model_KNN)
Evaluation Matrix K Nearest Neighbors
MAE Score:  169800.04452027066
MSE Score:  76775666245.47299
RMSE Score:  277084.2222961694
R2 :  0.45
Decision Tree
In [30]:
from sklearn.tree import DecisionTreeRegressor
In [31]:
model_desc_tree=DecisionTreeRegressor()
In [32]:
model_desc_tree.fit(X_train, y_train)
Out[32]:
DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort='deprecated',
                      random_state=None, splitter='best')
In [33]:
y_desc_tree = model_desc_tree.predict(X_test)
In [34]:
r2_model_desc_tree= (r2_score(y_test, y_desc_tree)).round(2)
MAE_model_desc_tree = mean_absolute_error(y_test, y_desc_tree)
MSE_model_desc_tree= mean_squared_error(y_test, y_desc_tree)
RMSE_mode_desc_tree= np.sqrt(MSE_model_desc_tree)

print("Evaluation Matrix Decission Tree")
print("MAE Score: ", MAE_model_desc_tree)
print("MSE Score: ", MSE_model_desc_tree)
print("RMSE Score: ", RMSE_mode_desc_tree)
print("R2 : ", r2_model_desc_tree)
Evaluation Matrix Decission Tree
MAE Score:  109260.97270256202
MSE Score:  40759547200.46099
RMSE Score:  201889.93833388772
R2 :  0.71
Random Forest
In [35]:
from sklearn.ensemble import RandomForestRegressor
In [36]:
model_RF = RandomForestRegressor()
In [37]:
model_RF.fit(X_train, y_train)
Out[37]:
RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
In [38]:
y_model_RF = model_RF.predict(X_test)
In [39]:
r2_model_RF= (r2_score(y_test, y_model_RF)).round(2)
MAE_model_RF = mean_absolute_error(y_test, y_model_RF)
MSE_model_RF= mean_squared_error(y_test, y_model_RF)
RMSE_mode_RF= np.sqrt(MSE_model_RF)

print("Evaluation Matrix Random Forest")
print("MAE Score: ", MAE_model_RF)
print("MSE Score: ", MSE_model_RF)
print("RMSE Score: ", RMSE_mode_RF)
print("R2 : ", r2_model_RF)
Evaluation Matrix Random Forest
MAE Score:  80077.08343222871
MSE Score:  27062934123.440372
RMSE Score:  164508.15822761002
R2 :  0.81
In [40]:
data = {
    "Linear Regression" : [MAE_model_linreg, MSE_model_linreg, RMSE_model_linreg, r2_model_Linreg],
    "K Nearest Neighbors" : [MAE_model_KNN, MSE_model_KNN, RMSE_model_KNN, r2_model_KNN],
    "Decission Tree" : [MAE_model_desc_tree, MSE_model_desc_tree, RMSE_mode_desc_tree, r2_model_desc_tree],
    "Random Forest" : [MAE_model_RF, MSE_model_RF, RMSE_mode_RF, r2_model_RF]
}

pd.DataFrame(data=data, index=['MAE', 'MSE', 'RMSE', 'R2'])
Out[40]:
Linear Regression	K Nearest Neighbors	Decission Tree	Random Forest
MAE	1.235623e+05	1.698000e+05	1.092610e+05	8.007708e+04
MSE	4.282110e+10	7.677567e+10	4.075955e+10	2.706293e+10
RMSE	2.069326e+05	2.770842e+05	2.018899e+05	1.645082e+05
R2	6.900000e-01	4.500000e-01	7.100000e-01	8.100000e-01
Polynomial
In [41]:
from sklearn.preprocessing import PolynomialFeatures
In [42]:
apoli = PolynomialFeatures(degree=5, include_bias = False)
In [43]:
apoli.fit_transform(X_train,y_train)
Out[43]:
array([[ 4.00000000e+00,  2.50000000e+00,  2.99000000e+03, ...,
        -4.07385845e+09,  1.04988759e+10, -2.70570019e+10],
       [ 3.00000000e+00,  2.25000000e+00,  1.23000000e+03, ...,
        -4.16230954e+09,  1.06900269e+10, -2.74551119e+10],
       [ 4.00000000e+00,  2.75000000e+00,  4.43000000e+03, ...,
        -4.11983908e+09,  1.05495250e+10, -2.70137922e+10],
       ...,
       [ 3.00000000e+00,  2.50000000e+00,  2.12000000e+03, ...,
        -4.13153939e+09,  1.05740235e+10, -2.70625456e+10],
       [ 1.00000000e+00,  7.50000000e-01,  3.80000000e+02, ...,
        -4.12633122e+09,  1.06304672e+10, -2.73867576e+10],
       [ 4.00000000e+00,  2.50000000e+00,  3.13000000e+03, ...,
        -4.08690235e+09,  1.05311888e+10, -2.71369188e+10]])
In [44]:
df.head()
Out[44]:
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long
0	221900.0	3	1.00	1180	5650	1.0	0	0	3	7	1180	0	1955	0	98178	47.5112	-122.257
1	538000.0	3	2.25	2570	7242	2.0	0	0	3	7	2170	400	1951	1991	98125	47.7210	-122.319
2	180000.0	2	1.00	770	10000	1.0	0	0	3	6	770	0	1933	0	98028	47.7379	-122.233
3	604000.0	4	3.00	1960	5000	1.0	0	0	5	7	1050	910	1965	0	98136	47.5208	-122.393
4	510000.0	3	2.00	1680	8080	1.0	0	0	3	8	1680	0	1987	0	98074	47.6168	-122.045
In [47]:
poli = apoli.fit_transform(df[['price']])
In [49]:
x_poly = df.drop(columns='price')  ### Features 
y_poly = df['price'] ### Target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .80, random_state = 42)
In [57]:
model_poly=model_linreg.fit(x_poly,y_poly)

y_linreg_poly=model_poly.predict(x_poly)
In [58]:
r2_linreg_poly= r2_score(y_poly, y_linreg_poly)
MAE_linreg_poly = mean_absolute_error(y_poly, y_linreg_poly)
MSE_linreg_poly= mean_squared_error(y_poly, y_linreg_poly)
RMSE_linreg_poly= np.sqrt(MSE_linreg_poly)

print("Evaluation Matrix Linear Regression- Polynomial")
print("MAE Score: ", MAE_linreg_poly)
print("MSE Score: ", MSE_linreg_poly)
print("RMSE Score: ", RMSE_linreg_poly)
print("R2 : ", r2_linreg_poly)
Evaluation Matrix Linear Regression- Polynomial
MAE Score:  126148.38213053216
MSE Score:  40586297407.84716
RMSE Score:  201460.41151513407
R2 :  0.698861410204793
In [59]:
data = {
    "Linear Regression" : [MAE_model_linreg, MSE_model_linreg, RMSE_model_linreg, r2_model_Linreg],
    "K Nearest Neighbors" : [MAE_model_KNN, MSE_model_KNN, RMSE_model_KNN, r2_model_KNN],
    "Decission Tree" : [MAE_model_desc_tree, MSE_model_desc_tree, RMSE_mode_desc_tree, r2_model_desc_tree],
    "Random Forest" : [MAE_model_RF, MSE_model_RF, RMSE_mode_RF, r2_model_RF],
    "Linear Regression - polynomial" : [MAE_linreg_poly, MSE_linreg_poly, RMSE_linreg_poly, r2_linreg_poly]
}

pd.DataFrame(data=data, index=['MAE', 'MSE', 'RMSE', 'R2'])
Out[59]:
Linear Regression	K Nearest Neighbors	Decission Tree	Random Forest	Linear Regression - polynomial
MAE	1.235623e+05	1.698000e+05	1.092610e+05	8.007708e+04	1.261484e+05
MSE	4.282110e+10	7.677567e+10	4.075955e+10	2.706293e+10	4.058630e+10
RMSE	2.069326e+05	2.770842e+05	2.018899e+05	1.645082e+05	2.014604e+05
R2	6.900000e-01	4.500000e-01	7.100000e-01	8.100000e-01	6.988614e-01
In [ ]:
Random Forest Regressor is commended in this case, because its R2 Score is the highest among the others.
