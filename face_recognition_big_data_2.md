

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import sys
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sys.path.append("/kaggle/input/mtcnngithub/mtcnn-master/") # To work around having MTCNN on Kaggle
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import glob

import cv2
from mtcnn import MTCNN
from sklearn.svm import SVC

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-5-8bf88c7c318a> in <module>
          9 import glob
         10 
    ---> 11 import cv2
         12 from mtcnn import MTCNN
         13 from sklearn.svm import SVC
    

    ModuleNotFoundError: No module named 'cv2'





```python
# Getting one image as an example:
from IPython.display import Image as PImage
ben_afflec = '/kaggle/input/5-celebrity-faces-dataset/data/train/ben_afflek/httpwwwallposterscomimagesPostersPFjpg.jpg'
PImage(ben_afflec)
```




![jpeg](output_3_0.jpeg)




```python
# Convert the image into pixels using matplotlib imread function, and printing an example. 
img_pix = plt.imread(ben_afflec)  
#print(img_pix) 
```


```python
# building the detector object from MTCNN
detector = MTCNN()

# detecting the face in the image and printing the returned value:
face = detector.detect_faces(img_pix) # if more than one face, it will return a list of multiple elements. 
print(face)
```

    [{'box': [50, 42, 132, 186], 'confidence': 1.0, 'keypoints': {'left_eye': (86, 116), 'right_eye': (146, 121), 'nose': (113, 150), 'mouth_left': (83, 180), 'mouth_right': (135, 185)}}]
    


```python
# Using the returned value above, I will create a rectangle around the face, and show it in the image
from matplotlib.patches import Rectangle  

x, y, x2, y2 = face[0]['box']  # extracting the coordinates
rectangle = Rectangle((x, y), x2, y2, color='green', fill=False)  # creating a rectangle using the coords
print(rectangle)
```

    Rectangle(xy=(50, 42), width=132, height=186, angle=0)
    


```python
# Showing the image with the rectangle
ax = plt.subplot()  # setting up the subplot
ax.imshow(img_pix)  # showing the image, we have to use the flattened image
ax.add_patch(rectangle)  # adding the rectangle created above
plt.show()
```


![png](output_7_0.png)



```python
# let's crop the face out of the image, and print it. 
box = face[0]['box']
img_cropped = img_pix[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
plt.imshow(img_cropped)
plt.show()

```


![png](output_8_0.png)



```python
# Another way, probably a better way:
from PIL import Image

box = face[0]['box']
# Coords: (left, top, right, bottom)
coords = (box[0],box[1],box[0]+box[2],box[1]+box[3])

cropped_image = Image.open(ben_afflec).crop(coords)
cropped_image
```




![png](output_9_0.png)




```python
# We can also convert it to grayscale and resize it, 
#  which are important steps for future classification. 
cropped_image_gs = cropped_image.convert('L').resize((100,100))
display(cropped_image_gs)
```


![png](output_10_0.png)



```python
# to flatten the image into an array format:
np.asarray(cropped_image_gs).flatten()
```




    array([ 6,  6,  6, ..., 16, 16, 15], dtype=uint8)




```python
# Now, let's scale everything up to be able to read all the images. 

# Step 0, set up the detector, images, labels, desired image size, and error list
detector = MTCNN()
images = []
labels = []
size = (100,100)
errors = []
ages = []

# Step 1, get all the training files
files = glob.glob('/kaggle/input/5-celebrity-faces-dataset/**/**/*.*')



# Step 2, loop over the files
for f in files:
    try: 
        # Step 3, get the person's name
        name = f.split('/')[-2]
        # Step 4, read the image to pixel mode
        img_pix = plt.imread(f)  
        # Step 5, detect the face
        face = detector.detect_faces(img_pix) # Assuming we only have one face per picture
        box = face[0]['box']
        # Step 6, crop the face out of the image
        coords = (box[0],box[1],box[0]+box[2],box[1]+box[3])  # Coords: (left, top, right, bottom)
        cropped_image = Image.open(f).crop(coords)
        # Step 7, convert the image to grayscale and resize it. 
        gs = cropped_image.convert('L').resize(size)
        # Step 7a, Optional: to label the age information manually.
        """display(Image.open(f))
        age = input("What is this person's age?")
        ages.append(age)"""
        # Step 8, flatten the image to array/pixel format and append it to the images list, also append its label
        images.append(np.asarray(gs).flatten())
        labels.append(name)
    except:
        errors.append(f)

# Step 9, create a df for all the images and their labels
df = pd.DataFrame(images)
df['target'] = labels
    
    
```


```python
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>9991</th>
      <th>9992</th>
      <th>9993</th>
      <th>9994</th>
      <th>9995</th>
      <th>9996</th>
      <th>9997</th>
      <th>9998</th>
      <th>9999</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>23</td>
      <td>27</td>
      <td>27</td>
      <td>31</td>
      <td>35</td>
      <td>35</td>
      <td>27</td>
      <td>27</td>
      <td>14</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>mindy_kaling</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>85</td>
      <td>77</td>
      <td>73</td>
      <td>72</td>
      <td>91</td>
      <td>102</td>
      <td>102</td>
      <td>90</td>
      <td>77</td>
      <td>...</td>
      <td>139</td>
      <td>139</td>
      <td>142</td>
      <td>142</td>
      <td>140</td>
      <td>140</td>
      <td>140</td>
      <td>139</td>
      <td>139</td>
      <td>mindy_kaling</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>7</td>
      <td>13</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>mindy_kaling</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>30</td>
      <td>42</td>
      <td>37</td>
      <td>37</td>
      <td>25</td>
      <td>25</td>
      <td>31</td>
      <td>...</td>
      <td>11</td>
      <td>20</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>mindy_kaling</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>20</td>
      <td>56</td>
      <td>56</td>
      <td>58</td>
      <td>73</td>
      <td>73</td>
      <td>67</td>
      <td>67</td>
      <td>82</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>9</td>
      <td>mindy_kaling</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 10001 columns</p>
</div>




```python
# Let's make the labels as numbers in a cool way:
labels = {}
ID = 0
numerical_label = []
for i in set(df.target):
    labels[i] = ID
    ID +=1

for i in df.target:
    numerical_label.append(labels[i])

df['target'] = numerical_label
#df['age'] = ages

#df.to_csv('facial_data.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>9991</th>
      <th>9992</th>
      <th>9993</th>
      <th>9994</th>
      <th>9995</th>
      <th>9996</th>
      <th>9997</th>
      <th>9998</th>
      <th>9999</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>23</td>
      <td>27</td>
      <td>27</td>
      <td>31</td>
      <td>35</td>
      <td>35</td>
      <td>27</td>
      <td>27</td>
      <td>14</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88</td>
      <td>85</td>
      <td>77</td>
      <td>73</td>
      <td>72</td>
      <td>91</td>
      <td>102</td>
      <td>102</td>
      <td>90</td>
      <td>77</td>
      <td>...</td>
      <td>139</td>
      <td>139</td>
      <td>142</td>
      <td>142</td>
      <td>140</td>
      <td>140</td>
      <td>140</td>
      <td>139</td>
      <td>139</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>7</td>
      <td>13</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>30</td>
      <td>42</td>
      <td>37</td>
      <td>37</td>
      <td>25</td>
      <td>25</td>
      <td>31</td>
      <td>...</td>
      <td>11</td>
      <td>20</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>20</td>
      <td>56</td>
      <td>56</td>
      <td>58</td>
      <td>73</td>
      <td>73</td>
      <td>67</td>
      <td>67</td>
      <td>82</td>
      <td>...</td>
      <td>8</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
      <td>9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 10001 columns</p>
</div>



# Different Classifications models (ensemble technique)


```python
# seperating images and labels
y=df["target"]
X=df.drop(["target"],axis=1)

# split data into training and test set
# this data will be used for all the below analysis
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size = 0.25)


```

### PCA


```python

# finding n_components
pca=PCA(n_components=87).fit(X_train)
components=range(1,88)
# finding the cumulative explained variance ratio
cumulative_var=np.cumsum(pca.explained_variance_ratio_)
# finding the explained variance ratio for each component
var=pca.explained_variance_ratio_
x=pd.DataFrame((zip(components,cumulative_var)),columns=['Number of components','cumulative explained variance ratio'])
y=pd.DataFrame((zip(components,var)),columns=['Number of components','explained variance ratio'])
pd.set_option('display.max_rows', None)
print(x)
```

        Number of components  cumulative explained variance ratio
    0                      1                             0.310697
    1                      2                             0.420363
    2                      3                             0.481915
    3                      4                             0.523840
    4                      5                             0.564490
    5                      6                             0.595013
    6                      7                             0.622262
    7                      8                             0.646216
    8                      9                             0.665630
    9                     10                             0.683770
    10                    11                             0.699644
    11                    12                             0.713315
    12                    13                             0.726450
    13                    14                             0.738856
    14                    15                             0.751050
    15                    16                             0.762145
    16                    17                             0.772646
    17                    18                             0.782801
    18                    19                             0.792073
    19                    20                             0.800731
    20                    21                             0.808801
    21                    22                             0.816249
    22                    23                             0.823603
    23                    24                             0.830168
    24                    25                             0.836598
    25                    26                             0.842958
    26                    27                             0.849216
    27                    28                             0.855263
    28                    29                             0.860785
    29                    30                             0.866109
    30                    31                             0.871269
    31                    32                             0.876344
    32                    33                             0.881032
    33                    34                             0.885626
    34                    35                             0.890099
    35                    36                             0.894401
    36                    37                             0.898599
    37                    38                             0.902622
    38                    39                             0.906570
    39                    40                             0.910445
    40                    41                             0.914146
    41                    42                             0.917750
    42                    43                             0.921192
    43                    44                             0.924492
    44                    45                             0.927658
    45                    46                             0.930723
    46                    47                             0.933758
    47                    48                             0.936711
    48                    49                             0.939636
    49                    50                             0.942392
    50                    51                             0.945038
    51                    52                             0.947668
    52                    53                             0.950185
    53                    54                             0.952559
    54                    55                             0.954898
    55                    56                             0.957167
    56                    57                             0.959386
    57                    58                             0.961513
    58                    59                             0.963597
    59                    60                             0.965634
    60                    61                             0.967649
    61                    62                             0.969627
    62                    63                             0.971465
    63                    64                             0.973294
    64                    65                             0.975048
    65                    66                             0.976788
    66                    67                             0.978486
    67                    68                             0.980094
    68                    69                             0.981628
    69                    70                             0.983082
    70                    71                             0.984496
    71                    72                             0.985852
    72                    73                             0.987160
    73                    74                             0.988453
    74                    75                             0.989695
    75                    76                             0.990925
    76                    77                             0.992115
    77                    78                             0.993263
    78                    79                             0.994334
    79                    80                             0.995358
    80                    81                             0.996273
    81                    82                             0.997165
    82                    83                             0.997938
    83                    84                             0.998675
    84                    85                             0.999359
    85                    86                             1.000000
    86                    87                             1.000000
    


```python
# bar chart
plt.bar(y['Number of components'], y['explained variance ratio'])
plt.title('Variance explained by each additional component')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
```




    Text(0, 0.5, 'Explained variance ratio')




![png](output_20_1.png)



```python
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Variance explained additional components cumulatively')
plt.xlabel('Number of components')
plt.ylabel('Cunmulative explained variance ratio')
plt.show()
```


![png](output_21_0.png)


### We can see that around 57 components explains 96% of the data. This is a good number of components to work with


```python
# setting n_components to 57
pca=PCA(n_components=57).fit(X_train)

#Displaying Eigenfaces
def show_eigenfaces(pca):

    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(100, 100), cmap='gray')
        ax.set_title("PC " + str(i+1))
    plt.show()

show_eigenfaces(pca)
```


![png](output_23_0.png)



```python
# training on pca values
x_train_pca=pca.transform(X_train)
x_test_pca=pca.transform(X_test)
clf= SVC(kernel='rbf',class_weight='balanced',C=10000)
clf=clf.fit(x_train_pca,y_train)
y_pred_pca=clf.predict(x_test_pca)

accuracy_score(y_pred_pca,y_test)
#a=pd.DataFrame(y_pred_pca,y_test)
#print(a)
```




    0.6551724137931034



### KNN Classification


```python
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                         weights='uniform')




```python
#finding accuracy for difffernt k values to find the best k
results = []
for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    results.append({
        'k': k,
        'accuracy': accuracy_score(y_test, knn.predict(X_test))
    })
# Convert results to a pandas data frame
results = pd.DataFrame(results)
max_k=(results.loc[(results['accuracy']==(results['accuracy'].max()))])['k'].max()
print('K='+ str(max_k) +' provides the best accuracy')
print()
print(results)
```

    K=2 provides the best accuracy
    
         k  accuracy
    0    1  0.551724
    1    2  0.620690
    2    3  0.517241
    3    4  0.517241
    4    5  0.413793
    5    6  0.379310
    6    7  0.448276
    7    8  0.413793
    8    9  0.448276
    9   10  0.448276
    10  11  0.482759
    11  12  0.448276
    12  13  0.413793
    13  14  0.448276
    14  15  0.413793
    15  16  0.413793
    16  17  0.413793
    17  18  0.413793
    18  19  0.448276
    19  20  0.448276
    20  21  0.517241
    21  22  0.482759
    22  23  0.448276
    23  24  0.448276
    24  25  0.448276
    25  26  0.379310
    26  27  0.379310
    27  28  0.413793
    28  29  0.413793
    


```python
# fit the data to knn model
knn = KNeighborsClassifier(n_neighbors=max_k).fit(X_train, y_train)
y_pred_knn=knn.predict(X_test)

# find accuracy
accuracy_score(y_test, y_pred_knn)
```




    0.6206896551724138



### Linear Discriminant Analysis


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred_lda=clf.predict(X_test)
accuracy_score(y_pred_lda,y_test)
```




    0.5172413793103449



### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred_logit=clf.predict(X_test)
accuracy_score(y_pred_logit,y_test)
```




    0.5517241379310345



### Support Vector Machine


```python
clf = SVC(kernel='rbf',C=10000)
clf = clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
accuracy_score(y_pred_svc,y_test)
```




    0.6206896551724138




```python
import statistics
# create a dataframe with all predictions
predictions = pd.DataFrame({'PCA':y_pred_pca,'KNN':y_pred_knn,'LDA':y_pred_lda,'Logit':y_pred_logit,'SVC':y_pred_svc},columns=['PCA','KNN', 'LDA','Logit','SVC'])
# find mode of each row
print(predictions)
y_pred_mode=predictions.mode(axis=1)
predictions.insert(5,"Mode",y_pred_mode)
print(predictions)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-479-97a8990b1c2d> in <module>
          4 # find mode of each row
          5 y_pred_mode=predictions.mode(axis=1)
    ----> 6 predictions.insert(5,"Mode",y_pred_mode)
          7 print(predictions)
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
       3494         self._ensure_valid_index(value)
       3495         value = self._sanitize_column(column, value, broadcast=False)
    -> 3496         self._data.insert(loc, column, value, allow_duplicates=allow_duplicates)
       3497 
       3498     def assign(self, **kwargs) -> "DataFrame":
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/internals/managers.py in insert(self, loc, item, value, allow_duplicates)
       1179         new_axis = self.items.insert(loc, item)
       1180 
    -> 1181         block = make_block(values=value, ndim=self.ndim, placement=slice(loc, loc + 1))
       1182 
       1183         for blkno, count in _fast_count_smallints(self._blknos[loc:]):
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/internals/blocks.py in make_block(values, placement, klass, ndim, dtype)
       3039         values = DatetimeArray._simple_new(values, dtype=dtype)
       3040 
    -> 3041     return klass(values, ndim=ndim, placement=placement)
       3042 
       3043 
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/internals/blocks.py in __init__(self, values, placement, ndim)
        123         if self._validate_ndim and self.ndim and len(self.mgr_locs) != len(self.values):
        124             raise ValueError(
    --> 125                 f"Wrong number of items passed {len(self.values)}, "
        126                 f"placement implies {len(self.mgr_locs)}"
        127             )
    

    ValueError: Wrong number of items passed 2, placement implies 1



```python
# predict mode with the test data
accuracy_score(y_pred_mode,y_test)
```


```python
print(kaggle/input/5-celebrity-faces-dataset/train/madonna/httpssmediacacheakpinimgcomxffabffabbbcfbceaedjpg.jpg)
```
