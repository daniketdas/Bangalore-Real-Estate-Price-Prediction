# Bangalore-Real-Estate-Price-Prediction
In this project, I collected data from Kaggle and conducted thorough data cleaning, feature engineering, and outlier removal processes. Subsequently, I developed a linear regression model to predict real estate prices in Bangalore. The project aimed to provide a reliable tool for estimating property values, facilitating informed decision-making in the Bangalore real estate market.
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1bec34db",
   "metadata": {},
   "source": [
    "# ||Real Estate Price Prediction: A Machine Learning Project||"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "253eece3",
   "metadata": {},
   "source": [
    "## Objective:\n",
    "### To develop a robust machine learning model capable of accurately predicting property prices in Bangalore. The objective encompasses the exploration, preprocessing, feature engineering, and training phases to create a predictive model that can assist homebuyers, real estate professionals, and stakeholders in making informed decisions regarding property investments in the Bangalore real estate market. The ultimate goal is to achieve high predictive performance, generalizability, and interpretability, thereby providing valuable insights into the dynamic and heterogeneous nature of property pricing trends in Bangalore."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "159b8aa1",
   "metadata": {},
   "source": [
    "## Data source:\n",
    "### Dataset used here in this project has been taken from Kaggle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6d1344e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import matplotlib\n",
    "matplotlib.rcParams[\"figure.figsize\"]= (20,10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "72e76bdf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing the csv file\n",
    "df=pd.read_csv(\"banglore.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2e28020a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>area_type</th>\n",
       "      <th>availability</th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>society</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>balcony</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>19-Dec</td>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>Coomee</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Plot  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>Theanmp</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>Soiewre</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              area_type   availability                  location       size  \\\n",
       "0  Super built-up  Area         19-Dec  Electronic City Phase II      2 BHK   \n",
       "1            Plot  Area  Ready To Move          Chikka Tirupathi  4 Bedroom   \n",
       "2        Built-up  Area  Ready To Move               Uttarahalli      3 BHK   \n",
       "3  Super built-up  Area  Ready To Move        Lingadheeranahalli      3 BHK   \n",
       "4  Super built-up  Area  Ready To Move                  Kothanur      2 BHK   \n",
       "\n",
       "   society total_sqft  bath  balcony   price  \n",
       "0  Coomee        1056   2.0      1.0   39.07  \n",
       "1  Theanmp       2600   5.0      3.0  120.00  \n",
       "2      NaN       1440   2.0      3.0   62.00  \n",
       "3  Soiewre       1521   3.0      1.0   95.00  \n",
       "4      NaN       1200   2.0      1.0   51.00  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b11f303c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13320, 9)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "53fa4a55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Dropping some column which we don't want to include in our analysis.\n",
    "df1=df.drop([\"area_type\",\"society\",\"balcony\",\"availability\"], axis=1)\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee31addc",
   "metadata": {},
   "source": [
    "## Data Cleaning:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0d7bc463",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location       1\n",
       "size          16\n",
       "total_sqft     0\n",
       "bath          73\n",
       "price          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for nan values\n",
    "df1.isnull().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "900d0d99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      0\n",
       "size          0\n",
       "total_sqft    0\n",
       "bath          0\n",
       "price         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2=df1.dropna() #dropping all rows with nan values.\n",
    "df2.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "19144a67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13246, 5)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "13449143",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['2 BHK', '4 Bedroom', '3 BHK', '4 BHK', '6 Bedroom', '3 Bedroom',\n",
       "       '1 BHK', '1 RK', '1 Bedroom', '8 Bedroom', '2 Bedroom',\n",
       "       '7 Bedroom', '5 BHK', '7 BHK', '6 BHK', '5 Bedroom', '11 BHK',\n",
       "       '9 BHK', '9 Bedroom', '27 BHK', '10 Bedroom', '11 Bedroom',\n",
       "       '10 BHK', '19 BHK', '16 BHK', '43 Bedroom', '14 BHK', '8 BHK',\n",
       "       '12 Bedroom', '13 BHK', '18 Bedroom'], dtype=object)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the size column.\n",
    "df2[\"size\"].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee0acf1c",
   "metadata": {},
   "source": [
    "So, we can see that size is mention as BHK type in some cases and in other cases it's as Bedroom. As this may create problems in our analysis we have to write them in same format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6dfe5908",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Aniket Das\\AppData\\Local\\Temp\\ipykernel_47656\\2456293048.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df2[\"BHK\"]= df2[\"size\"].apply(lambda x: int(x.split(\" \")[0]))\n"
     ]
    }
   ],
   "source": [
    "#Creating a new column as 'BHK'.\n",
    "df2[\"BHK\"]= df2[\"size\"].apply(lambda x: int(x.split(\" \")[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7655822c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price  BHK\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00    3\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00    3\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00    2"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "3d3400f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  3,  6,  1,  8,  7,  5, 11,  9, 27, 10, 19, 16, 43, 14, 12,\n",
       "       13, 18], dtype=int64)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[\"BHK\"].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dae44d89",
   "metadata": {},
   "source": [
    "Intrestingly we can see some preoperties with more than 10 rooms."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6f7f82ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[(df2[\"BHK\"])>10].value_counts().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d45af2c3",
   "metadata": {},
   "source": [
    "So more 12 properties are listed with more than 10 bedrooms, which seems like an error."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6cb6016f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[\"total_sqft\"].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e4815fe",
   "metadata": {},
   "source": [
    "So few of the values in \"total_sqft\" column are in range instead of float datatype."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "135f6d5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def is_float(x):\n",
    "    try:\n",
    "        float(x)\n",
    "    except:\n",
    "        return False\n",
    "    return True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0a887a68",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>Yelahanka</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2100 - 2850</td>\n",
       "      <td>4.0</td>\n",
       "      <td>186.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>Hebbal</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>3067 - 8156</td>\n",
       "      <td>4.0</td>\n",
       "      <td>477.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>137</th>\n",
       "      <td>8th Phase JP Nagar</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1042 - 1105</td>\n",
       "      <td>2.0</td>\n",
       "      <td>54.005</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>165</th>\n",
       "      <td>Sarjapur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1145 - 1340</td>\n",
       "      <td>2.0</td>\n",
       "      <td>43.490</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>KR Puram</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1015 - 1540</td>\n",
       "      <td>2.0</td>\n",
       "      <td>56.800</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>410</th>\n",
       "      <td>Kengeri</td>\n",
       "      <td>1 BHK</td>\n",
       "      <td>34.46Sq. Meter</td>\n",
       "      <td>1.0</td>\n",
       "      <td>18.500</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>549</th>\n",
       "      <td>Hennur Road</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1195 - 1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>63.770</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>648</th>\n",
       "      <td>Arekere</td>\n",
       "      <td>9 Bedroom</td>\n",
       "      <td>4125Perch</td>\n",
       "      <td>9.0</td>\n",
       "      <td>265.000</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>661</th>\n",
       "      <td>Yelahanka</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1120 - 1145</td>\n",
       "      <td>2.0</td>\n",
       "      <td>48.130</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>672</th>\n",
       "      <td>Bettahalsoor</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>3090 - 5002</td>\n",
       "      <td>4.0</td>\n",
       "      <td>445.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               location       size      total_sqft  bath    price  BHK\n",
       "30            Yelahanka      4 BHK     2100 - 2850   4.0  186.000    4\n",
       "122              Hebbal      4 BHK     3067 - 8156   4.0  477.000    4\n",
       "137  8th Phase JP Nagar      2 BHK     1042 - 1105   2.0   54.005    2\n",
       "165            Sarjapur      2 BHK     1145 - 1340   2.0   43.490    2\n",
       "188            KR Puram      2 BHK     1015 - 1540   2.0   56.800    2\n",
       "410             Kengeri      1 BHK  34.46Sq. Meter   1.0   18.500    1\n",
       "549         Hennur Road      2 BHK     1195 - 1440   2.0   63.770    2\n",
       "648             Arekere  9 Bedroom       4125Perch   9.0  265.000    9\n",
       "661           Yelahanka      2 BHK     1120 - 1145   2.0   48.130    2\n",
       "672        Bettahalsoor  4 Bedroom     3090 - 5002   4.0  445.000    4"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2[~df2[\"total_sqft\"].apply(is_float)].head(10) #to get all the data that can not be converted to float."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abe1cc93",
   "metadata": {},
   "source": [
    "So from this above table we can see that not only \"total_sqft\" column has values in range but also it includes some records that are in object datatype (eg. 4125Perch)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e28301ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Replacing all the records that are in range with their average.\n",
    "def total_sqft_to_num(c):\n",
    "    tokens=c.split(\"-\")\n",
    "    if len(tokens)==2:\n",
    "        return(float(tokens[0])+float(tokens[1]))/2\n",
    "    try:\n",
    "        return float(c)\n",
    "    except:\n",
    "        None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1b301277",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creatig a copied dataframe of df2 and applying \"total_sqft_to_num\" function on total_sqft column.\n",
    "df3=df2.copy()\n",
    "df3[\"total_sqft\"]=df3[\"total_sqft\"].apply(total_sqft_to_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "87d5b6cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  BHK\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3\n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3\n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d6908056",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      Yelahanka\n",
       "size              4 BHK\n",
       "total_sqft       2475.0\n",
       "bath                4.0\n",
       "price             186.0\n",
       "BHK                   4\n",
       "Name: 30, dtype: object"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.loc[30]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a08162f",
   "metadata": {},
   "source": [
    "All the values in range are now replaced with their average."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d43a33ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      Kengeri\n",
       "size            1 BHK\n",
       "total_sqft        NaN\n",
       "bath              1.0\n",
       "price            18.5\n",
       "BHK                 1\n",
       "Name: 410, dtype: object"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.loc[410]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74201b73",
   "metadata": {},
   "source": [
    "Records with units like 34.46Sq. Meter, 4125Perch are now showing NaN values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "e50b40bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location       0\n",
       "size           0\n",
       "total_sqft    46\n",
       "bath           0\n",
       "price          0\n",
       "BHK            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c05e6aba",
   "metadata": {},
   "source": [
    "Total 46 NaN values in 'total_sqft' column, indicating 46 records with units replaced with NaN values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2bbd26c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creating a new column named price_per_sqft as it is an important factor in real estate market.\n",
    "df4=df3.copy()\n",
    "df4[\"price_per_sqft\"]=(df4[\"price\"]*100000)/df4[\"total_sqft\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "36ef0db7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "      <td>3699.810606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "      <td>4615.384615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "      <td>4305.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "      <td>6245.890861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "      <td>4250.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  BHK  \\\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2   \n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4   \n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3   \n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3   \n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2   \n",
       "\n",
       "   price_per_sqft  \n",
       "0     3699.810606  \n",
       "1     4615.384615  \n",
       "2     4305.555556  \n",
       "3     6245.890861  \n",
       "4     4250.000000  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "315084f4",
   "metadata": {},
   "source": [
    "## Dimentionality Reduction:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3ac19b9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli', ...,\n",
       "       '12th cross srinivas nagar banshankari 3rd stage',\n",
       "       'Havanur extension', 'Abshot Layout'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#unique locations\n",
    "df4.location=df4.location.apply(lambda x: x.strip())\n",
    "df4[\"location\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "17ed00de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1293"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df4[\"location\"].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6bcf85f2",
   "metadata": {},
   "source": [
    "Total 1293 unique locations are listed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "279b8e3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Deleting all the leading spaces or any spaces at the end in location.\n",
    "df4.location=df4.location.apply(lambda x: x.strip())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "4ca84b13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "1 Annasandrapalya      1\n",
       "Kudlu Village,         1\n",
       "Kumbhena Agrahara      1\n",
       "Kuvempu Layout         1\n",
       "LIC Colony             1\n",
       "                    ... \n",
       "Thanisandra          236\n",
       "Kanakpura Road       266\n",
       "Electronic City      304\n",
       "Sarjapur  Road       392\n",
       "Whitefield           535\n",
       "Name: location, Length: 1293, dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "location_stats= df4.groupby(\"location\")[\"location\"].agg(\"count\").sort_values()\n",
    "location_stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "0e09eaa6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1052"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for locations that have less than 10 data points.\n",
    "len(location_stats[location_stats<=10])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0a896fc",
   "metadata": {},
   "source": [
    "So 1052 locations are there which have 10 or less than 10 data points."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "ad3c4c63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "242"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Converting all these 1052 locations to a general category called 'other'.\n",
    "location_less_than_10 = location_stats[location_stats<=10]\n",
    "df4.location=df4.location.apply(lambda x: \"other\" if x in location_less_than_10 else x )\n",
    "len(df4[\"location\"].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cd52638",
   "metadata": {},
   "source": [
    "So, now we only have 242 unique data points in location."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "5a9d4bee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13246, 7)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa195fb9",
   "metadata": {},
   "source": [
    "## Outlier Removal:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "928fbf6a",
   "metadata": {},
   "source": [
    "Firstly, we will look into all those datasets where the total_sqft per bhk is less than 300 (as there might be some cases where per bedroom square ft is very less like 6 bhk with 600 sqft, that means 100 sqft per bedroom, this is suspicious and will drop all those records as anomalies keeping 300 per bhk sqft as threshold)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f148a038",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>other</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>370.0</td>\n",
       "      <td>6</td>\n",
       "      <td>36274.509804</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>HSR Layout</td>\n",
       "      <td>8 Bedroom</td>\n",
       "      <td>600.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>200.0</td>\n",
       "      <td>8</td>\n",
       "      <td>33333.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>58</th>\n",
       "      <td>Murugeshpalya</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1407.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>6</td>\n",
       "      <td>10660.980810</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>Devarachikkanahalli</td>\n",
       "      <td>8 Bedroom</td>\n",
       "      <td>1350.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>85.0</td>\n",
       "      <td>8</td>\n",
       "      <td>6296.296296</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70</th>\n",
       "      <td>other</td>\n",
       "      <td>3 Bedroom</td>\n",
       "      <td>500.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>3</td>\n",
       "      <td>20000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13277</th>\n",
       "      <td>other</td>\n",
       "      <td>7 Bedroom</td>\n",
       "      <td>1400.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>218.0</td>\n",
       "      <td>7</td>\n",
       "      <td>15571.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13279</th>\n",
       "      <td>other</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>6</td>\n",
       "      <td>10833.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13281</th>\n",
       "      <td>Margondanahalli</td>\n",
       "      <td>5 Bedroom</td>\n",
       "      <td>1375.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>125.0</td>\n",
       "      <td>5</td>\n",
       "      <td>9090.909091</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13303</th>\n",
       "      <td>Vidyaranyapura</td>\n",
       "      <td>5 Bedroom</td>\n",
       "      <td>774.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>5</td>\n",
       "      <td>9043.927649</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13311</th>\n",
       "      <td>Ramamurthy Nagar</td>\n",
       "      <td>7 Bedroom</td>\n",
       "      <td>1500.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>250.0</td>\n",
       "      <td>7</td>\n",
       "      <td>16666.666667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>744 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                  location       size  total_sqft  bath  price  BHK  \\\n",
       "9                    other  6 Bedroom      1020.0   6.0  370.0    6   \n",
       "45              HSR Layout  8 Bedroom       600.0   9.0  200.0    8   \n",
       "58           Murugeshpalya  6 Bedroom      1407.0   4.0  150.0    6   \n",
       "68     Devarachikkanahalli  8 Bedroom      1350.0   7.0   85.0    8   \n",
       "70                   other  3 Bedroom       500.0   3.0  100.0    3   \n",
       "...                    ...        ...         ...   ...    ...  ...   \n",
       "13277                other  7 Bedroom      1400.0   7.0  218.0    7   \n",
       "13279                other  6 Bedroom      1200.0   5.0  130.0    6   \n",
       "13281      Margondanahalli  5 Bedroom      1375.0   5.0  125.0    5   \n",
       "13303       Vidyaranyapura  5 Bedroom       774.0   5.0   70.0    5   \n",
       "13311     Ramamurthy Nagar  7 Bedroom      1500.0   9.0  250.0    7   \n",
       "\n",
       "       price_per_sqft  \n",
       "9        36274.509804  \n",
       "45       33333.333333  \n",
       "58       10660.980810  \n",
       "68        6296.296296  \n",
       "70       20000.000000  \n",
       "...               ...  \n",
       "13277    15571.428571  \n",
       "13279    10833.333333  \n",
       "13281     9090.909091  \n",
       "13303     9043.927649  \n",
       "13311    16666.666667  \n",
       "\n",
       "[744 rows x 7 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4[df4.total_sqft/df4.BHK<300]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62c06942",
   "metadata": {},
   "source": [
    "So, total 744 such cases are there where total sq ft per bhk is less than 300."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "7b900b87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12502, 7)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5= df4[~(df4.total_sqft/df4.BHK<300)]\n",
    "df5.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "15ab0a02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     12456.000000\n",
       "mean       6308.502826\n",
       "std        4168.127339\n",
       "min         267.829813\n",
       "25%        4210.526316\n",
       "50%        5294.117647\n",
       "75%        6916.666667\n",
       "max      176470.588235\n",
       "Name: price_per_sqft, dtype: float64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking all the extreme cases of price_per_sqft.\n",
    "df5[\"price_per_sqft\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f04b8c4e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10241, 7)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Removing all existing outliers in \"price_per_sqft\" column.\n",
    "def remove_pps_outlier(df):\n",
    "    df_out=pd.DataFrame()\n",
    "    for key,subdf in df.groupby(\"location\"):\n",
    "        mean=np.mean(subdf.price_per_sqft)\n",
    "        std=np.std(subdf.price_per_sqft)\n",
    "        reduced_df=subdf[(subdf.price_per_sqft<=(mean+std)) & (subdf.price_per_sqft>(mean-std))]\n",
    "        df_out=pd.concat([df_out,reduced_df], ignore_index=True)\n",
    "    return df_out\n",
    "\n",
    "df6=remove_pps_outlier(df5)\n",
    "df6.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f34aca0",
   "metadata": {},
   "source": [
    "So, all the extreme cases 'in price_per_sqft' are removed."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "225ca751",
   "metadata": {},
   "source": [
    "### Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "31390571",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABNYAAANVCAYAAAC09nNHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABmE0lEQVR4nO3deZhdVZ0v7s9JVRJCJVUmYFKJSQMKOBAGBRVICCgzAoH4a3BohNbrFQloCDhg31a0FXAgAbUD9r02QZFGbQnIVVBoBhOG24BGwEYcGhGbDLaGFHUoMlSd3x/HVFKZs1NVp4b3fZ791D5rr7PPdyeb8vjJWnuVKpVKJQAAAADADhlS6wIAAAAAoD8SrAEAAABAAYI1AAAAAChAsAYAAAAABQjWAAAAAKAAwRoAAAAAFCBYAwAAAIACBGsAAAAAUIBgDQAAAAAKEKwBAHSD+fPnp1QqdW719fUZP3583vGOd+TXv/514fPuueeeOeecc7rlvb/73e9SKpUyf/78rb5vXb9SqZSbbrppk+OXXnppSqVS/vu//7tQXQAAA0V9rQsAABhIrrvuurzmNa/JSy+9lPvvvz+f+9zncs899+SXv/xlRo8evcPnW7BgQRobGwvVsvF7x48fnwcffDCvetWrtvscf/d3f5e3v/3tGTp0aKEaAAAGMiPWAAC60eTJk3PooYfmqKOOyt/93d/l4x//eJYvX55bbrml0Ple//rX71AQtrX3Dh8+PIceemhe/vKXb9f7TzzxxPznf/5nrr322kKfX0svvvhirUsAAAYBwRoAQA865JBDkiTLli3rbHvppZdy0UUX5aCDDkpTU1PGjBmTww47LLfeeusm7994OufOvHd7p4Ku89a3vjXHH398/uEf/iEvvPDCVvveeeedmT59eiZOnJhddtkle++9dz7wgQ9sdrrorbfemgMOOCDDhw/PK1/5ylx99dWd00s39I//+I+ZNm1axo4dm4aGhuy///75whe+kDVr1nTpd9RRR2Xy5Mn5yU9+ksMPPzy77rpr3vve927XNQIA7AxTQQEAetDTTz+dJNl3330721atWpU///nPufjii/OKV7wiq1evzl133ZUZM2bkuuuuy3ve854tnm9n3lvE5z//+bz+9a/PF7/4xXzmM5/ZYr/f/va3Oeyww/I//sf/SFNTU373u99lzpw5mTp1ah5//PHOqaR33HFHZsyYkWnTpuXb3/521q5dmy996UtdgscNz/mud70re+21V4YNG5af//zn+dznPpdf/vKX+ed//ucufZcsWZK/+Zu/yUc/+tFcdtllGTLEvx8DAD1PsAYA0I3a29uzdu3azmesffazn820adNy6qmndvZpamrKdddd1+U9Rx99dFasWJGrrrpqq+HYzry3iAMPPDDvete7MmfOnJx33nlpbm7ebL9zzz23c79SqeTwww/PUUcdlT322CO333575/V/8pOfzCte8Yr86Ec/yrBhw5IkJ5xwQvbcc89NzjlnzpzO/Y6OjhxxxBHZbbfd8rd/+7e58soruzyz7s9//nO++93v5q1vfWt3XDYAwHbxT3kAAN3o0EMPzdChQzNq1KiccMIJGT16dG699dbU13f998zvfve7mTJlSkaOHJn6+voMHTo0X//61/Pkk09u8zN25r1FfPazn82aNWvy6U9/eot9li9fnnPPPTeTJk3qrGmPPfZIks66yuVyHnnkkZx22mmdoVqSjBw5Mqeccsom5/zZz36WU089Nbvttlvq6uoydOjQvOc970l7e3t+9atfdek7evRooRoA0OsEawAA3egb3/hGHn744dx99935wAc+kCeffDLvfOc7u/S5+eabc8YZZ+QVr3hFbrjhhjz44IN5+OGH8973vjcvvfTSVs+/M+8tas8998x5552X//N//k9+/etfb3K8o6Mjxx13XG6++eZ89KMfzb/927/l3//93/PQQw8lSdra2pIkK1asSKVSybhx4zY5x8Ztv//973PEEUfkv/7rv3L11Vdn4cKFefjhh/OP//iPXc65zvjx47vlWgEAdoSpoAAA3ei1r31t54IFb3nLW9Le3p7/83/+T/71X/81/9//9/8lSW644Ybstdde+fa3v93lgf2rVq3a5vl35r0743/9r/+Vf/7nf84nPvGJ7Lfffl2OPfHEE/n5z3+e+fPn5+yzz+5s/81vftOl3+jRo1MqlTb7PLWlS5d2eX3LLbekXC7n5ptv7hz5liSLFy/ebH0bL3wAANAbjFgDAOhBX/jCFzJ69Oh88pOfTEdHR5JqCDRs2LAuYdDSpUs3u7LnxnbmvTtjt912y8c+9rH867/+a/793/99k5qSZPjw4V3av/a1r3V53dDQkEMOOSS33HJLVq9e3dne2tqa//t//+82z1mpVPK///f/3vmLAQDoJoI1AIAeNHr06FxyySV58sknc+ONNyZJTj755Dz11FM577zzcvfdd+f666/P1KlTt2s64868d2fNmjUrEyZMyO23396l/TWveU1e9apX5eMf/3j+5V/+JT/60Y9y/vnn5/vf//4m5/jMZz6T//qv/8rxxx+fW265Jd/73vdyzDHHZOTIkV3CwmOPPTbDhg3LO9/5ztx+++1ZsGBBjj/++KxYsaLHrxMAYHsJ1gAAetgFF1yQv/qrv8pnPvOZtLe352//9m9zxRVX5Pbbb89JJ52Uz3/+8/n4xz+ed73rXZt9/4aB0868d2ftuuuuufTSSzdpHzp0aG677bbsu++++cAHPpB3vvOdWb58ee66665N+p5wwgn53ve+lz/96U8588wzM3v27Jx++umZPn16Xvayl3X2e81rXpPvfe97WbFiRWbMmJELLrggBx10UL785S932/UAAOysUqVSqdS6CAAANm/MmDF573vfmy996Uu9+t7etGbNmhx00EF5xStekR//+Me1LgcAYLtZvAAAoA967LHH8sMf/jArVqzIYYcd1mvv7Q3ve9/7cuyxx2b8+PFZunRprr322jz55JO5+uqra10aAMAOEawBAPRBH/7wh/PLX/4yF198cWbMmNFr7+0NL7zwQi6++OL88Y9/zNChQ/OGN7whP/zhD3PMMcfUujQAgB1iKigAAAAAFGDxAgAAAAAoQLAGAAAAAAUI1gAAAACgAIsXJOno6Mhzzz2XUaNGpVQq1bocAAAAAGqkUqnkhRdeyIQJEzJkyNbHpAnWkjz33HOZNGlSrcsAAAAAoI949tlnM3HixK32EawlGTVqVJLqH1hjY2ONqwEAAACgVlpaWjJp0qTOvGhrBGtJ5/TPxsZGwRoAAAAA2/W4MIsXAAAAAEABgjUAAAAAKECwBgAAAAAFeMbadqpUKlm7dm3a29trXcqAUldXl/r6+u2atwwAAADQlwjWtsPq1auzZMmSvPjii7UuZUDaddddM378+AwbNqzWpQAAAABsN8HaNnR0dOTpp59OXV1dJkyYkGHDhhld1U0qlUpWr16dP/7xj3n66aezzz77ZMgQs5MBAACA/kGwtg2rV69OR0dHJk2alF133bXW5Qw4I0aMyNChQ/PMM89k9erV2WWXXWpdEgAAAMB2MTxoOxlJ1XP82QIAAAD9kUQDAAAAAAoQrAEAAABAAYI1Ol166aU56KCDttrnnHPOyWmnnbbVPnvuuWeuuuqqbqsLAAAAoC8SrPWitrZk2bLqz552+eWX541vfGNGjRqVsWPH5rTTTstTTz3V8x8MAAAAMEgI1nrBokXJjBnJyJFJc3P154wZyf3399xn3nfffZk5c2Yeeuih3HnnnVm7dm2OO+64lMvlnvtQAAAAgEFEsNbDrrkmmTYtue22pKOj2tbRUX19xBHJtdf2zOfecccdOeecc7LffvvlwAMPzHXXXZff//73efTRR7f53q997WuZNGlSdt111/z1X/91nn/++U36fOlLX8r48eOz2267ZebMmVmzZs0Wz3fdddelqakpd955585cEgAAAECfIljrQYsWJTNnJpVKsnZt12Nr11bbzzuvZ0eurbNy5cokyZgxY7ba7ze/+U2+853v5Lbbbssdd9yRxYsXZ+bMmV363HPPPfntb3+be+65J9dff33mz5+f+fPnb/Z8X/rSl3LxxRfnRz/6UY499thuuRYAAACAvkCw1oPmzEnq6rbep64umTu3Z+uoVCqZPXt2pk6dmsmTJ2+170svvZTrr78+Bx10UKZNm5avfOUruemmm7J06dLOPqNHj85Xv/rVvOY1r8nJJ5+ct73tbfm3f/u3Tc51ySWXZM6cObn33ntz6KGHdvt1AQAAANRSfa0LGKja2pJbb10//XNL1q5NFiyo9h8xomdqOf/88/PYY49l0aJF2+z7V3/1V5k4cWLn68MOOywdHR156qmn0tzcnCTZb7/9UrdBYjh+/Pg8/vjjXc5z5ZVXplwu55FHHskrX/nKbroSAAAAgL7DiLUe0tKy7VBtnY6Oav+ecMEFF+T73/9+7rnnni6B2fYqlUpdfibJ0KFDN+nTsdHFHnHEEWlvb893vvOdAlUDAAAA9H2CtR7S2JgM2c4/3SFDqv27U6VSyfnnn5+bb745d999d/baa6/tet/vf//7PPfcc52vH3zwwQwZMiT77rvvDn3+m970ptxxxx257LLL8sUvfnGH3gsAAADQHwjWesiIEcn06Un9Nibb1tcnp5/e/dNAZ86cmRtuuCE33nhjRo0alaVLl2bp0qVpa2vb6vt22WWXnH322fn5z3+ehQsX5kMf+lDOOOOMzmmgO+Kwww7L7bffns985jOZ29MPkgMAAADoZZ6x1oNmz05uuWXrfdrbkwsv7P7Pvuaaa5IkRx11VJf26667Luecc84W37f33ntnxowZOemkk/LnP/85J510UubNm1e4jilTpuQHP/hBTjrppNTV1eVDH/pQ4XMBAAAA9CWlSqVSqXURtdbS0pKmpqasXLkyjRvNyXzppZfy9NNPZ6+99souu+yyw+e+9trkvPOqq3+uXbu+vb6+GqrNm5ece+7OXkH/trN/xgAAAADdZWs50cZMBe1h556bLFxYnRa67plrQ4ZUXy9cKFQDAAAA6K9MBe0FU6ZUt7a26uqfjY3d/0w1AAAAAHqXYK0XjRghUAMAAAAYKEwFBQAAAIACBGsAAAAAUIBgDQAAAGCwKJeTUqm6lcu1rqbfE6wBAAAAQAGCNQAAAAAowKqgAAAAAAPZhlM+t7SfJA0NvVPPAGLEGp0uvfTSHHTQQVvtc8455+S0007bap8999wzV111VbfVBQAAAOyEkSPXb+PGrW8fN67rMXaYYG2Auuaaa3LAAQeksbExjY2NOeyww3L77bfXuiwAAACAAcNU0N5SLq9Pf1tbe3x45cSJE3PFFVdk7733TpJcf/31mT59en72s59lv/3269HPBgAAAPqQ1tb1++Xy+lFry5aZ/rmTjFgboE455ZScdNJJ2XfffbPvvvvmc5/7XEaOHJmHHnpom+/92te+lkmTJmXXXXfNX//1X+f555/fpM+XvvSljB8/PrvttltmzpyZNWvWbPF81113XZqamnLnnXfuzCUBAAAARTQ0dN221c52E6wNAu3t7bnppptSLpdz2GGHbbXvb37zm3znO9/JbbfdljvuuCOLFy/OzJkzu/S555578tvf/jb33HNPrr/++syfPz/z58/f7Pm+9KUv5eKLL86PfvSjHHvssd11SQAAAAA1ZypoT6rxqhuPP/54DjvssLz00ksZOXJkFixYkNe97nVbfc9LL72U66+/PhMnTkySfOUrX8nb3va2XHnllWlubk6SjB49Ol/96ldTV1eX17zmNXnb296Wf/u3f8v73//+Lue65JJLcv311+fee+/N/vvv3yPXCAAAAFArgrWetKUVNTZcgSNJKpUe+fhXv/rVWbx4cZ5//vl873vfy9lnn5377rtvq+HaX/3VX3WGakly2GGHpaOjI0899VRnsLbffvulrq6us8/48ePz+OOPdznPlVde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3/rWrFq1KkmydOnSDBs2LKNHj+5yznHjxmXp0qVb/MzLL788TU1NndukSZN67gIBAAAAGJBqOhU0SV796ldn8eLFef755/O9730vZ599du6777687nWvy5lnntnZb/LkyTnkkEOyxx575Ac/+EFmzJixxXNWKpWUSqUtHr/kkksye/bsztctLS3CNQAAAKD3tbau3y+X149aW7bM9M9+oObB2rBhw7L33nsnSQ455JA8/PDDufrqq/O1r31tk77jx4/PHnvskV//+tdJkubm5qxevTorVqzoMmpt+fLlOfzww7f4mcOHD8/w4cO7+UoAAAAAdtCWwrOGBsFaP1DzqaAbq1QqnVM9N/anP/0pzz77bMaPH58kOfjggzN06NDceeednX2WLFmSJ554YqvBGgAAAADsrJqOWPvEJz6RE088MZMmTcoLL7yQm266Kffee2/uuOOOtLa25tJLL83b3/72jB8/Pr/73e/yiU98IrvvvntOP/30JElTU1Pe97735aKLLspuu+2WMWPG5OKLL87+++/fuUooAAAAAPSEmgZry5Yty1lnnZUlS5akqakpBxxwQO64444ce+yxaWtry+OPP55vfOMbef755zN+/Pi85S1vybe//e2MGjWq8xxz585NfX19zjjjjLS1teXoo4/O/PnzU1dXV8MrAwAAANhBDQ1JpVLrKtgBpUrF31hLS0uampqycuXKNDY21rocAAAAAGpkR3KiPveMNQAAAADoDwRrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFCNYAAAAAoADBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAAAAAAoQrAEAAMBgVS4npVJ1K5drXQ3+PvodwRoAAAAAFCBYAwAAAIAC6mtdAAAAANCLNpxiuKX9JGlo6J16Bjt/H/2aYA0AAAAGk5EjN98+blzX15VKz9eCv49+zlRQAAAAACjAiDUAAAAYTFpb1++Xy+tHRi1bZrphLfj76NcEawAAADCYbCmsaWgQ5NSCv49+zVRQAAAAAChAsAYAAAAABZgKCgAAAINVQ4PVJvsSfx/9jhFrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAABgx5TLSalU3crlWlcDNSNYAwAAAIACBGsAAAAAUEB9rQsAAAAA+oENp3xuaT9JGhp6px7oAwRrAAAA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XfqsWrUqF1xwQXbfffc0NDTk1FNPzR/+8IfevJRB7ZprkmnTkttuSzo6qm0dHdXXRxyRXHttbesDAAAA6Ak1Ddbuu+++zJw5Mw899FDuvPPOrF27Nscdd1zKG/zL9Re+8IXMmTMnX/3qV/Pwww+nubk5xx57bF544YXOPrNmzcqCBQty0003ZdGiRWltbc3JJ5+c9vb2WlzWoLJoUTJzZlKpJGvXdj22dm21/bzzjFwDAAAABp5SpVKp1LqIdf74xz9m7Nixue+++zJt2rRUKpVMmDAhs2bNysc+9rEk1dFp48aNy+c///l84AMfyMqVK/Pyl7883/zmN3PmmWcmSZ577rlMmjQpP/zhD3P88cdv83NbWlrS1NSUlStXprGxsUevcaCZMaM6Mm3jUG1D9fXJ9OnJv/5r79UFQDfZcJpmuZyMG1fdX7YsaWhYf2zD/d44FwAA9JAdyYn61DPWVq5cmSQZM2ZMkuTpp5/O0qVLc9xxx3X2GT58eI488sg88MADSZJHH300a9as6dJnwoQJmTx5cmefja1atSotLS1dNnZcW1ty661bD9WS6vEFCyxoANAvjRy5flsXhCXV/Q2P9fa5AACgD+gzwVqlUsns2bMzderUTJ48OUmydOnSJMm4Db98/+X1umNLly7NsGHDMnr06C322djll1+epqamzm3SpEndfTmDQkvL+meqbUtHR7U/AAAAwEDRZ4K1888/P4899lj+5V/+ZZNjpVKpy+tKpbJJ28a21ueSSy7JypUrO7dnn322eOGDWGNjMmQ776AhQ6r9AehnWlvXb8uWrW9ftqzrsd4+FwAA9AF9Ili74IIL8v3vfz/33HNPJk6c2Nne3NycJJuMPFu+fHnnKLbm5uasXr06K1as2GKfjQ0fPjyNjY1dNnbciBHVZ6fV12+9X319cvrp1f4A9DMNDV23bbX31rkAAKAPqGmwVqlUcv755+fmm2/O3Xffnb322qvL8b322ivNzc258847O9tWr16d++67L4cffniS5OCDD87QoUO79FmyZEmeeOKJzj70nNmzk20tvtrenlx4Ye/UAwAAANBbtjHWqGfNnDkzN954Y2699daMGjWqc2RaU1NTRowYkVKplFmzZuWyyy7LPvvsk3322SeXXXZZdt1117zrXe/q7Pu+970vF110UXbbbbeMGTMmF198cfbff/8c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RU9xb9FT3Fv0FPcWPcW9RU/q7vtrWyPV1rF4AQAAAAAUIFgDAAAAgAIEa3Sr4cOH51Of+lSGDx9e61IYYNxb9BT3Fj3FvUVPcW/RU9xb9BT3Fj2p1veXxQsAAAAAoAAj1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYI1N/OQnP8kpp5ySCRMmpFQq5ZZbbuk8tmbNmnzsYx/L/vvvn4aGhkyYMCHvec978txzz3U5x6pVq3LBBRdk9913T0NDQ0499dT84Q9/6NJnxYoVOeuss9LU1JSmpqacddZZef7553vhCqmVrd1bG/vABz6QUqmUq666qku7e4vN2Z5768knn8ypp56apqamjBo1Koceemh+//vfdx53b7E527q3Wltbc/7552fixIkZMWJEXvva1+aaa67p0se9xeZcfvnleeMb35hRo0Zl7NixOe200/LUU0916VOpVHLppZdmwoQJGTFiRI466qj84he/6NLH/cXGtnVv+T5PUdvze2tDvs+zvbb33uqr3+cFa2yiXC7nwAMPzFe/+tVNjr344ov56U9/mr//+7/PT3/609x888351a9+lVNPPbVLv1mzZmXBggW56aabsmjRorS2tubkk09Oe3t7Z593vetdWbx4ce64447ccccdWbx4cc4666wevz5qZ2v31oZuueWW/L//9/8yYcKETY65t9icbd1bv/3tbzN16tS85jWvyb333puf//zn+fu///vssssunX3cW2zOtu6tCy+8MHfccUduuOGGPPnkk7nwwgtzwQUX5NZbb+3s495ic+67777MnDkzDz30UO68886sXbs2xx13XMrlcmefL3zhC5kzZ06++tWv5uGHH05zc3OOPfbYvPDCC5193F9sbFv3lu/zFLU9v7fW8X2eHbE991af/j5fga1IUlmwYMFW+/z7v/97JUnlmWeeqVQqlcrzzz9fGTp0aOWmm27q7PNf//VflSFDhlTuuOOOSqVSqfzHf/xHJUnloYce6uzz4IMPVpJUfvnLX3b/hdDnbOne+sMf/lB5xSteUXniiScqe+yxR2Xu3Lmdx9xbbI/N3Vtnnnlm5W/+5m+2+B73Fttjc/fWfvvtV/nMZz7Tpe0Nb3hD5X/9r/9VqVTcW2y/5cuXV5JU7rvvvkqlUql0dHRUmpubK1dccUVnn5deeqnS1NRUufbaayuVivuL7bPxvbU5vs9TxJbuLd/n2Vmbu7f68vd5I9bYaStXrkypVMrLXvayJMmjjz6aNWvW5LjjjuvsM2HChEyePDkPPPBAkuTBBx9MU1NT3vzmN3f2OfTQQ9PU1NTZh8Gno6MjZ511Vj7ykY9kv/322+S4e4siOjo68oMf/CD77rtvjj/++IwdOzZvfvObu0zpc29R1NSpU/P9738///Vf/5VKpZJ77rknv/rVr3L88ccncW+x/VauXJkkGTNmTJLk6aefztKlS7vcO8OHD8+RRx7ZeV+4v9geG99bW+rj+zw7anP3lu/zdIeN762+/n1esMZOeemll/Lxj38873rXu9LY2JgkWbp0aYYNG5bRo0d36Ttu3LgsXbq0s8/YsWM3Od/YsWM7+zD4fP7zn099fX0+9KEPbfa4e4sili9fntbW1lxxxRU54YQT8uMf/zinn356ZsyYkfvuuy+Je4vivvzlL+d1r3tdJk6cmGHDhuWEE07IvHnzMnXq1CTuLbZPpVLJ7NmzM3Xq1EyePDlJOv/ux40b16XvxveO+4ut2dy9tTHf5yliS/eW7/PsrM3dW339+3x94Xcy6K1ZsybveMc70tHRkXnz5m2zf6VSSalU6ny94f6W+jB4PProo7n66qvz05/+dIfvAfcWW9PR0ZEkmT59ei688MIkyUEHHZQHHngg1157bY488sgtvte9xbZ8+ctfzkMPPZTvf//72WOPPfKTn/wk5513XsaPH59jjjlmi+9zb7Gh888/P4899lgWLVq0ybGN74HtuS/cX6yztXsr8X2e4jZ3b/k+T3fY3L3V17/PG7FGIWvWrMkZZ5yRp59+OnfeeWfnv24lSXNzc1avXp0VK1Z0ec/y5cs7/9W1ubk5y5Yt2+S8f/zjHzf5l1kGh4ULF2b58uX5q7/6q9TX16e+vj7PPPNMLrroouy5555J3FsUs/vuu6e+vj6ve93rurS/9rWv7VxFyL1FEW1tbfnEJz6ROXPm5JRTTskBBxyQ888/P2eeeWa+9KUvJXFvsW0XXHBBvv/97+eee+7JxIkTO9ubm5uTZJN/Qd/43nF/sSVburfW8X2eorZ0b/k+z87a0r3V17/PC9bYYev+R/jXv/517rrrruy2225djh988MEZOnRo7rzzzs62JUuW5Iknnsjhhx+eJDnssMOycuXK/Pu//3tnn//3//5fVq5c2dmHweWss87KY489lsWLF3duEyZMyEc+8pH86Ec/SuLeophhw4bljW984yZLdv/qV7/KHnvskcS9RTFr1qzJmjVrMmRI169TdXV1nf+y6t5iSyqVSs4///zcfPPNufvuu7PXXnt1Ob7XXnulubm5y72zevXq3HfffZ33hfuLzdnWvZX4Pk8x27q3fJ+nqG3dW33++3zhZQ8YsF544YXKz372s8rPfvazSpLKnDlzKj/72c8qzzzzTGXNmjWVU089tTJx4sTK4sWLK0uWLOncVq1a1XmOc889tzJx4sTKXXfdVfnpT39aeetb31o58MADK2vXru3sc8IJJ1QOOOCAyoMPPlh58MEHK/vvv3/l5JNPrsUl00u2dm9tzsarCFUq7i02b1v31s0331wZOnRo5Z/+6Z8qv/71rytf+cpXKnV1dZWFCxd2nsO9xeZs69468sgjK/vtt1/lnnvuqfznf/5n5brrrqvssssulXnz5nWew73F5nzwgx+sNDU1Ve69994u36defPHFzj5XXHFFpampqXLzzTdXHn/88co73/nOyvjx4ystLS2dfdxfbGxb95bv8xS1Pb+3Nub7PNtje+6tvvx9XrDGJu65555Kkk22s88+u/L0009v9liSyj333NN5jra2tsr5559fGTNmTGXEiBGVk08+ufL73/++y+f86U9/qrz73e+ujBo1qjJq1KjKu9/97sqKFSt692LpVVu7tzZnc/9D7N5ic7bn3vr6179e2XvvvSu77LJL5cADD6zccsstXc7h3mJztnVvLVmypHLOOedUJkyYUNlll10qr371qytXXnllpaOjo/Mc7i02Z0vfp6677rrOPh0dHZVPfepTlebm5srw4cMr06ZNqzz++ONdzuP+YmPburd8n6eo7fm9tTHf59ke23tv9dXv86W/XAQAAAAAsAM8Yw0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFCNYAAAAAoADBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAKAf2nPPPXPVVVf16mfef//92X///TN06NCcdtppvfrZAAB9kWANAGAnlEqlrW7nnHPONt9/yy239EqtO2v27Nk56KCD8vTTT2f+/Pm59NJLc9BBB9W6LACAmqmvdQEAAP3ZkiVLOve//e1v55Of/GSeeuqpzrYRI0bUoqwe8dvf/jbnnntuJk6c2OOftWbNmgwdOrTHPwcAYGcYsQYAsBOam5s7t6amppRKpS5tN954Y171qldl2LBhefWrX51vfvObne/dc889kySnn356SqVS5+vf/va3mT59esaNG5eRI0fmjW98Y+66664dquvee+/Nm970pjQ0NORlL3tZpkyZkmeeeabz+BVXXJFx48Zl1KhRed/73pePf/zjWxx99rvf/S6lUil/+tOf8t73vjelUinz58/Ppz/96fz85z/vHJ03f/78zb7/4YcfzrHHHpvdd989TU1NOfLII/PTn/60S59SqZRrr70206dPT0NDQz772c8mSW677bYcfPDB2WWXXfLKV74yn/70p7N27drO982ZMyf7779/GhoaMmnSpJx33nlpbW3doT8rAICiBGsAAD1kwYIF+fCHP5yLLrooTzzxRD7wgQ/kb//2b3PPPfckqQZOSXLddddlyZIlna9bW1tz0kkn5a677srPfvazHH/88TnllFPy+9//frs+d+3atTnttNNy5JFH5rHHHsuDDz6Y//k//2dKpVKS5Dvf+U4+9alP5XOf+1weeeSRjB8/PvPmzdvi+SZNmpQlS5aksbExV111VZYsWZIzzzwzF110Ufbbb78sWbKks21zXnjhhZx99tlZuHBhHnrooeyzzz456aST8sILL3Tp96lPfSrTp0/P448/nve+97350Y9+lL/5m7/Jhz70ofzHf/xHvva1r2X+/Pn53Oc+1/meIUOG5Mtf/nKeeOKJXH/99bn77rvz0Y9+dLv+nAAAdlapUqlUal0EAMBAMH/+/MyaNSvPP/98kmTKlCnZb7/98k//9E+dfc4444yUy+X84Ac/SFIdqbVgwYJtLgaw33775YMf/GDOP//8JNXRbrNmzcqsWbM26fvnP/85u+22W+69994ceeSRmxw//PDDc+CBB+aaa67pbDv00EPz0ksvZfHixVus4WUve1muuuqqzufGXXrppbnlllu2+p7NaW9vz+jRo3PjjTfm5JNPTlL9c5g1a1bmzp3b2W/atGk58cQTc8kll3S23XDDDfnoRz+a5557brPn/u53v5sPfvCD+e///u8dqgkAoAgj1gAAesiTTz6ZKVOmdGmbMmVKnnzyya2+r1wu56Mf/Whe97rX5WUve1lGjhyZX/7yl9s9Ym3MmDE555xzOke6XX311V2eBffkk0/msMMO6/KejV93p+XLl+fcc8/Nvvvum6ampjQ1NaW1tXWT6znkkEO6vH700Ufzmc98JiNHjuzc3v/+92fJkiV58cUXkyT33HNPjj322LziFa/IqFGj8p73vCd/+tOfUi6Xe+x6AADWEawBAPSgddMv16lUKpu0bewjH/lIvve97+Vzn/tcFi5cmMWLF2f//ffP6tWrt/tzr7vuujz44IM5/PDD8+1vfzv77rtvHnrooULXsLPOOeecPProo7nqqqvywAMPZPHixdltt902uZ6GhoYurzs6OvLpT386ixcv7twef/zx/PrXv84uu+ySZ555JieddFImT56c733ve3n00Ufzj//4j0mqix8AAPQ0q4ICAPSQ1772tVm0aFHe8573dLY98MADee1rX9v5eujQoWlvb+/yvoULF+acc87J6aefnqT6zLXf/e53O/z5r3/96/P6178+l1xySQ477LDceOONOfTQQ/Pa1742Dz30UJe6ioRuw4YN26T2zVm4cGHmzZuXk046KUny7LPPbtdUzTe84Q156qmnsvfee2/2+COPPJK1a9fmyiuvzJAh1X8v/s53vrMDVwAAsHMEawAAPeQjH/lIzjjjjLzhDW/I0Ucfndtuuy0333xzlxU+99xzz/zbv/1bpkyZkuHDh2f06NHZe++9c/PNN+eUU05JqVTK3//936ejo2O7P/fpp5/OP/3TP+XUU0/NhAkT8tRTT+VXv/pVZ5D24Q9/OGeffXYOOeSQTJ06Nd/61rfyi1/8Iq985St36Pr23HPPPP3001m8eHEmTpyYUaNGZfjw4Zv023vvvfPNb34zhxxySFpaWvKRj3wkI0aM2Ob5P/nJT+bkk0/OpEmT8td//dcZMmRIHnvssTz++OP57Gc/m1e96lVZu3ZtvvKVr+SUU07J/fffn2uvvXaHrgEAYGeYCgoA0ENOO+20XH311fniF7+Y/fbbL1/72tdy3XXX5aijjursc+WVV+bOO+/MpEmT8vrXvz5JMnfu3IwePTqHH354TjnllBx//PF5wxvesN2fu+uuu+aXv/xl3v72t2fffffN//yf/zPnn39+PvCBDyRJzjzzzHzyk5/Mxz72sRx88MF55pln8sEPfnCHr+/tb397TjjhhLzlLW/Jy1/+8vzLv/zLZvv98z//c1asWJHXv/71Oeuss/KhD30oY8eO3eb5jz/++Pzf//t/c+edd+aNb3xjDj300MyZMyd77LFHkuSggw7KnDlz8vnPfz6TJ0/Ot771rVx++eU7fB0AAEVZFRQAgMIrfAIADGZGrAEAAABAAYI1AAAAACjAVFAAAAAAKMCINQAAAAAoQLAGAAAAAAUI1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYA0AAAAAChCsAQAAAEAB/z/PSGS7cffEdwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1500x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def sctter_plot(df,location):\n",
    "    bhk2=df[(df.location==location) & (df.BHK==2)]\n",
    "    bhk3=df[(df.location==location) & (df.BHK==3)]\n",
    "    matplotlib.rcParams[\"figure.figsize\"]=(15,10)\n",
    "    plt.scatter(bhk2.total_sqft,bhk2.price, color=\"blue\", label=\"2 bhk\", s=50)\n",
    "    plt.scatter(bhk3.total_sqft,bhk3.price, marker=\"+\", color=\"red\", label=\"3 bhk\", s=50)\n",
    "    plt.xlabel(\"Total sq ft area\")\n",
    "    plt.ylabel(\"Price\")\n",
    "    plt.title(location)\n",
    "    plt.legend()\n",
    "    \n",
    "sctter_plot(df6, \"Rajaji Nagar\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "470a0545",
   "metadata": {},
   "source": [
    "We can see here that in few cases for same sqft area 2 bhk costs more than 3 bhk, this could be for various reasons but as this might affect our model, we need to clean this up."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea1f7670",
   "metadata": {},
   "source": [
    "### Removing all the records  where for the same price and location 3 bhk price is less than 2 bhk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "89b3b295",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7329, 7)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def bhk_outlier_removal(df):\n",
    "    exclude_indices=np.array([])\n",
    "    for location, location_df in df.groupby(\"location\"):\n",
    "        bhk_stats={}\n",
    "        for BHK, bhk_df in location_df.groupby(\"BHK\"):\n",
    "            bhk_stats[BHK]={\n",
    "                \"mean\":np.mean(bhk_df.price_per_sqft),\n",
    "                \"Standad_dev\":np.std(bhk_df.price_per_sqft),\n",
    "                \"count\":bhk_df.shape[0]\n",
    "            }\n",
    "        for BHK, bhk_df in location_df.groupby(\"BHK\"):\n",
    "            stats=bhk_stats.get(BHK-1)\n",
    "            if stats and stats[\"count\"]>5:\n",
    "                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats[\"mean\"])].index.values)\n",
    "    return df.drop(exclude_indices,axis=\"index\")\n",
    "\n",
    "df7=bhk_outlier_removal(df6)\n",
    "df7.shape\n",
    "    \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "433d7dd1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABNYAAANVCAYAAAC09nNHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABjTUlEQVR4nO3de5yWZYE//s/DDCAOzAS6MBCsWmoH8VBaqSBank1F6bfaYU23to1AC9Es2++WtaV2ELRasP1+W7CspdpE81tSuB4CD99Vi9TW7LCmtnLYLWScceQw8/z+eJbB4czNzDzDzPv9et2vee7rvp77ue7hCp8+XIdSuVwuBwAAAADYJQOq3QAAAAAA2BMJ1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFCNYAALrA/PnzUyqVOo7a2tqMHj0673znO/Ob3/ym8H3333//XHTRRV3y3t///vcplUqZP3/+dt+3sV6pVMqCBQu2uH7VVVelVCrlv//7vwu1CwCgr6itdgMAAPqSefPm5bWvfW1eeuml3Hffffnc5z6Xu+++O7/61a8yfPjwXb7fwoULU19fX6gtm7939OjReeCBB/LqV796p+/xt3/7t3nHO96RgQMHFmoDAEBfZsQaAEAXGj9+fI4++uiccMIJ+du//dt8/OMfz6pVq3LrrbcWut8b3vCGXQrCtvfewYMH5+ijj86f/dmf7dT7Tz/99PzHf/xHbrzxxkKfX00vvvhitZsAAPQDgjUAgG501FFHJUlWrlzZUfbSSy/lsssuyxFHHJGGhoaMGDEixxxzTG677bYt3r/5dM7dee/OTgXd6G1ve1tOPfXU/P3f/31eeOGF7dZdvHhxJk+enLFjx2avvfbKgQcemA9+8INbnS5622235bDDDsvgwYPzqle9KjfccEPH9NKX+4d/+IdMmjQpI0eOTF1dXQ499NB84QtfyPr16zvVO+GEEzJ+/Pj89Kc/zbHHHpu9994773vf+3bqGQEAdoepoAAA3eipp55Kkhx88MEdZWvXrs2f/vSnXH755XnlK1+ZdevW5c4778yUKVMyb968vPe9793m/XbnvUV8/vOfzxve8IZ88YtfzGc+85lt1vvd736XY445Jn/913+dhoaG/P73v8+sWbMyceLEPPbYYx1TSRctWpQpU6Zk0qRJ+c53vpMNGzbkS1/6Uqfg8eX3fPe7350DDjgggwYNyi9+8Yt87nOfy69+9av80z/9U6e6y5cvz1/+5V/miiuuyNVXX50BA/z7MQDQ/QRrAABdqK2tLRs2bOhYY+2zn/1sJk2alLPPPrujTkNDQ+bNm9fpPSeeeGJWr16d66+/frvh2O68t4jDDz887373uzNr1qxMmzYtjY2NW603derUjtflcjnHHntsTjjhhOy333654447Op7/k5/8ZF75ylfmxz/+cQYNGpQkOe2007L//vtvcc9Zs2Z1vG5vb89xxx2XffbZJ3/1V3+V6667rtOadX/605/yve99L29729u64rEBAHaKf8oDAOhCRx99dAYOHJhhw4bltNNOy/Dhw3Pbbbeltrbzv2d+73vfy4QJEzJ06NDU1tZm4MCB+frXv54nnnhih5+xO+8t4rOf/WzWr1+fT3/609uss2rVqkydOjXjxo3raNN+++2XJB3tamlpycMPP5xzzjmnI1RLkqFDh+ass87a4p4///nPc/bZZ2efffZJTU1NBg4cmPe+971pa2vLr3/96051hw8fLlQDAHqcYA0AoAt94xvfyEMPPZS77rorH/zgB/PEE0/kXe96V6c6t9xyS84777y88pWvzM0335wHHnggDz30UN73vvflpZde2u79d+e9Re2///6ZNm1a/s//+T/5zW9+s8X19vb2nHLKKbnllltyxRVX5F//9V/zb//2b3nwwQeTJK2trUmS1atXp1wuZ9SoUVvcY/OyZ555Jscdd1z+8z//MzfccEOWLFmShx56KP/wD//Q6Z4bjR49ukueFQBgV5gKCgDQhV73utd1bFjw1re+NW1tbfk//+f/5F/+5V/y//1//1+S5Oabb84BBxyQ73znO50W7F+7du0O7787790d/+t//a/80z/9Uz7xiU/kkEMO6XTt8ccfzy9+8YvMnz8/F154YUf5b3/72071hg8fnlKptNX11FasWNHp/NZbb01LS0tuueWWjpFvSbJs2bKttm/zjQ8AAHqCEWsAAN3oC1/4QoYPH55PfvKTaW9vT1IJgQYNGtQpDFqxYsVWd/bc3O68d3fss88++djHPpZ/+Zd/yb/9279t0aYkGTx4cKfyr33ta53O6+rqctRRR+XWW2/NunXrOsqbm5vzf//v/93hPcvlcv73//7fu/8wAABdRLAGANCNhg8fniuvvDJPPPFEvv3tbydJzjzzzDz55JOZNm1a7rrrrtx0002ZOHHiTk1n3J337q4ZM2ZkzJgxueOOOzqVv/a1r82rX/3qfPzjH88///M/58c//nEuvvji/OAHP9jiHp/5zGfyn//5nzn11FNz66235vvf/35OOumkDB06tFNYePLJJ2fQoEF517velTvuuCMLFy7MqaeemtWrV3f7cwIA7CzBGgBAN7vkkkvy53/+5/nMZz6Ttra2/NVf/VWuvfba3HHHHTnjjDPy+c9/Ph//+Mfz7ne/e6vvf3ngtDvv3V177713rrrqqi3KBw4cmNtvvz0HH3xwPvjBD+Zd73pXVq1alTvvvHOLuqeddlq+//3v549//GPOP//8zJw5M+eee24mT56cV7ziFR31Xvva1+b73/9+Vq9enSlTpuSSSy7JEUcckS9/+ctd9jwAALurVC6Xy9VuBAAAWzdixIi8733vy5e+9KUefW9PWr9+fY444oi88pWvzE9+8pNqNwcAYKfZvAAAoBd69NFH86Mf/SirV6/OMccc02Pv7Qnvf//7c/LJJ2f06NFZsWJFbrzxxjzxxBO54YYbqt00AIBdIlgDAOiFPvKRj+RXv/pVLr/88kyZMqXH3tsTXnjhhVx++eX5r//6rwwcODBvfOMb86Mf/SgnnXRStZsGALBLTAUFAAAAgAJsXgAAAAAABQjWAAAAAKAAwRoAAAAAFGDzgiTt7e157rnnMmzYsJRKpWo3BwAAAIAqKZfLeeGFFzJmzJgMGLD9MWmCtSTPPfdcxo0bV+1mAAAAANBLPPvssxk7dux26wjWkgwbNixJ5RdWX19f5dYAAAAAUC1NTU0ZN25cR160PYK1pGP6Z319vWANAAAAgJ1aLszmBQAAAABQgGANAAAAAAoQrAEAAABAAdZY20nlcjkbNmxIW1tbtZvSp9TU1KS2tnan5i0DAAAA9CaCtZ2wbt26LF++PC+++GK1m9In7b333hk9enQGDRpU7aYAAAAA7DTB2g60t7fnqaeeSk1NTcaMGZNBgwYZXdVFyuVy1q1bl//6r//KU089lYMOOigDBpidDAAAAOwZBGs7sG7durS3t2fcuHHZe++9q92cPmfIkCEZOHBgnn766axbty577bVXtZsEAAAAsFMMD9pJRlJ1H79bAAAAYE8k0QAAAACAAgRrAAAAAFCAYI0OV111VY444ojt1rnoootyzjnnbLfO/vvvn+uvv77L2gUAAADQGwnWelBra7JyZeVnd7vmmmvypje9KcOGDcvIkSNzzjnn5Mknn+z+DwYAAADoJwRrPWDp0mTKlGTo0KSxsfJzypTkvvu67zPvvffeTJ8+PQ8++GAWL16cDRs25JRTTklLS0v3fSgAAABAPyJY62Zz5yaTJiW33560t1fK2tsr58cdl9x4Y/d87qJFi3LRRRflkEMOyeGHH5558+blmWeeySOPPLLD937ta1/LuHHjsvfee+cv/uIv8vzzz29R50tf+lJGjx6dffbZJ9OnT8/69eu3eb958+aloaEhixcv3p1HAgAAAOhVBGvdaOnSZPr0pFxONmzofG3Dhkr5tGndO3JtozVr1iRJRowYsd16v/3tb/Pd7343t99+exYtWpRly5Zl+vTpnercfffd+d3vfpe77747N910U+bPn5/58+dv9X5f+tKXcvnll+fHP/5xTj755C55FgAAAIDeQLDWjWbNSmpqtl+npiaZPbt721EulzNz5sxMnDgx48eP327dl156KTfddFOOOOKITJo0KV/5yleyYMGCrFixoqPO8OHD89WvfjWvfe1rc+aZZ+btb397/vVf/3WLe1155ZWZNWtW7rnnnhx99NFd/lwAAAAA1VRb7Qb0Va2tyW23bZr+uS0bNiQLF1bqDxnSPW25+OKL8+ijj2bp0qU7rPvnf/7nGTt2bMf5Mccck/b29jz55JNpbGxMkhxyyCGpeVliOHr06Dz22GOd7nPdddelpaUlDz/8cF71qld10ZMAAAAA9B5GrHWTpqYdh2obtbdX6neHSy65JD/4wQ9y9913dwrMdlapVOr0M0kGDhy4RZ32zR72uOOOS1tbW7773e8WaDUAAABA7ydY6yb19cmAnfztDhhQqd+VyuVyLr744txyyy256667csABB+zU+5555pk899xzHecPPPBABgwYkIMPPniXPv/Nb35zFi1alKuvvjpf/OIXd+m9AAAAAHsCwVo3GTIkmTw5qd3BZNva2uTcc7t+Guj06dNz880359vf/naGDRuWFStWZMWKFWltbd3u+/baa69ceOGF+cUvfpElS5bkwx/+cM4777yOaaC74phjjskdd9yRz3zmM5nd3QvJAQAAAPQwa6x1o5kzk1tv3X6dtrbk0ku7/rPnzp2bJDnhhBM6lc+bNy8XXXTRNt934IEHZsqUKTnjjDPypz/9KWeccUbmzJlTuB0TJkzID3/4w5xxxhmpqanJhz/84cL3AgAAAOhNSuVyuVztRlRbU1NTGhoasmbNmtRvNifzpZdeylNPPZUDDjgge+211y7f+8Ybk2nTKrt/btiwqby2thKqzZmTTJ26u0+wZ9vd3zEAAABAV9leTrQ5U0G72dSpyZIllWmhG9dcGzCgcr5kiVANAAAAYE9lKmgPmDChcrS2Vnb/rK/v+jXVAAAAAOhZgrUeNGSIQA0AAACgrzAVFAAAAAAKEKwBAAAAQAGCNQAAAID+oqUlKZUqR0tLtVuzxxOsAQAAAEABgjUAAAAAKMCuoAAAAAB92cunfG7rdZLU1fVMe/oQI9bocNVVV+WII47Ybp2LLroo55xzznbr7L///rn++uu7rF0AAADAbhg6dNMxatSm8lGjOl9jlwnW+qi5c+fmsMMOS319ferr63PMMcfkjjvuqHazAAAAAPoMU0F7SkvLpvS3ubnbh1eOHTs21157bQ488MAkyU033ZTJkyfn5z//eQ455JBu/WwAAACgF2lu3vS6pWXTqLWVK03/3E1GrPVRZ511Vs4444wcfPDBOfjgg/O5z30uQ4cOzYMPPrjD937ta1/LuHHjsvfee+cv/uIv8vzzz29R50tf+lJGjx6dffbZJ9OnT8/69eu3eb958+aloaEhixcv3p1HAgAAAIqoq+t87KicnSZY6wfa2tqyYMGCtLS05Jhjjtlu3d/+9rf57ne/m9tvvz2LFi3KsmXLMn369E517r777vzud7/L3XffnZtuuinz58/P/Pnzt3q/L33pS7n88svz4x//OCeffHJXPRIAAABA1ZkK2p2qvOvGY489lmOOOSYvvfRShg4dmoULF+b1r3/9dt/z0ksv5aabbsrYsWOTJF/5ylfy9re/Pdddd10aGxuTJMOHD89Xv/rV1NTU5LWvfW3e/va351//9V/zgQ98oNO9rrzyytx000255557cuihh3bLMwIAAABUi2CtO21rR42X78CRJOVyt3z8a17zmixbtizPP/98vv/97+fCCy/Mvffeu91w7c///M87QrUkOeaYY9Le3p4nn3yyI1g75JBDUlNT01Fn9OjReeyxxzrd57rrrktLS0se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Fy7cbp1/+7d/KycpP/300+VyuVx+/vnnywMHDiwvWLCgo85//ud/lgcMGFBetGhRuVwul//93/+9nKT84IMPdtR54IEHyknKv/rVr7r+Qeh1ttW3/vCHP5Rf+cpXlh9//PHyfvvtV549e3bHNX2LnbG1vnX++eeX//Iv/3Kb79G32Blb61uHHHJI+TOf+Uynsje+8Y3l//W//le5XNa32HmrVq0qJynfe++95XK5XG5vby83NjaWr7322o46L730UrmhoaF84403lstl/Yuds3nf2hrf5yliW33L93l219b6Vm/+Pm/EGrttzZo1KZVKecUrXpEkeeSRR7J+/fqccsopHXXGjBmT8ePH5/7770+SPPDAA2loaMhb3vKWjjpHH310GhoaOurQ/7S3t+eCCy7IRz/60RxyyCFbXNe3KKK9vT0//OEPc/DBB+fUU0/NyJEj85a3vKXTlD59i6ImTpyYH/zgB/nP//zPlMvl3H333fn1r3+dU089NYm+xc5bs2ZNkmTEiBFJkqeeeiorVqzo1HcGDx6c448/vqNf6F/sjM371rbq+D7Prtpa3/J9nq6wed/q7d/nBWvslpdeeikf//jH8+53vzv19fVJkhUrVmTQoEEZPnx4p7qjRo3KihUrOuqMHDlyi/uNHDmyow79z+c///nU1tbmwx/+8Fav61sUsWrVqjQ3N+faa6/Naaedlp/85Cc599xzM2XKlNx7771J9C2K+/KXv5zXv/71GTt2bAYNGpTTTjstc+bMycSJE5PoW+yccrmcmTNnZuLEiRk/fnySdPzZjxo1qlPdzfuO/sX2bK1vbc73eYrYVt/yfZ7dtbW+1du/z9cWfif93vr16/POd74z7e3tmTNnzg7rl8vllEqljvOXv95WHfqPRx55JDfccEN+9rOf7XIf0LfYnvb29iTJ5MmTc+mllyZJjjjiiNx///258cYbc/zxx2/zvfoWO/LlL385Dz74YH7wgx9kv/32y09/+tNMmzYto0ePzkknnbTN9+lbvNzFF1+cRx99NEuXLt3i2uZ9YGf6hf7FRtvrW4nv8xS3tb7l+zxdYWt9q7d/nzdijULWr1+f8847L0899VQWL17c8a9bSdLY2Jh169Zl9erVnd6zatWqjn91bWxszMqVK7e473/9139t8S+z9A9LlizJqlWr8ud//uepra1NbW1tnn766Vx22WXZf//9k+hbFLPvvvumtrY2r3/96zuVv+51r+vYRUjfoojW1tZ84hOfyKxZs3LWWWflsMMOy8UXX5zzzz8/X/rSl5LoW+zYJZdckh/84Ae5++67M3bs2I7yxsbGJNniX9A37zv6F9uyrb61ke/zFLWtvuX7PLtrW32rt3+fF6yxyzb+R/g3v/lN7rzzzuyzzz6drh955JEZOHBgFi9e3FG2fPnyPP744zn22GOTJMccc0zWrFmTf/u3f+uo8//+3//LmjVrOurQv1xwwQV59NFHs2zZso5jzJgx+ehHP5of//jHSfQtihk0aFDe9KY3bbFl969//evst99+SfQtilm/fn3Wr1+fAQM6f52qqanp+JdVfYttKZfLufjii3PLLbfkrrvuygEHHNDp+gEHHJDGxsZOfWfdunW59957O/qF/sXW7KhvJb7PU8yO+pbv8xS1o77V67/PF972gD7rhRdeKP/85z8v//znPy8nKc+aNav885//vPz000+X169fXz777LPLY8eOLS9btqy8fPnyjmPt2rUd95g6dWp57Nix5TvvvLP8s5/9rPy2t72tfPjhh5c3bNjQUee0004rH3bYYeUHHnig/MADD5QPPfTQ8plnnlmNR6aHbK9vbc3muwiVy/oWW7ejvnXLLbeUBw4cWP7Hf/zH8m9+85vyV77ylXJNTU15yZIlHffQt9iaHfWt448/vnzIIYeU77777vJ//Md/lOfNm1fea6+9ynPmzOm4h77F1nzoQx8qNzQ0lO+5555O36defPHFjjrXXnttuaGhoXzLLbeUH3vssfK73vWu8ujRo8tNTU0ddfQvNrejvuX7PEXtzN9bm/N9np2xM32rN3+fF6yxhbvvvrucZIvjwgsvLD/11FNbvZakfPfdd3fco7W1tXzxxReXR4wYUR4yZEj5zDPPLD/zzDOdPuePf/xj+T3veU952LBh5WHDhpXf8573lFevXt2zD0uP2l7f2pqt/YdY32JrdqZvff3rXy8feOCB5b322qt8+OGHl2+99dZO99C32Jod9a3ly5eXL7roovKYMWPKe+21V/k1r3lN+brrriu3t7d33EPfYmu29X1q3rx5HXXa29vLn/rUp8qNjY3lwYMHlydNmlR+7LHHOt1H/2JzO+pbvs9T1M78vbU53+fZGTvbt3rr9/nS/zwEAAAAALALrLEGAAAAAAUI1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgCwB9p///1z/fXX9+hn3nfffTn00EMzcODAnHPOOT362QAAvZFgDQBgN5RKpe0eF1100Q7ff+utt/ZIW3fXzJkzc8QRR+Spp57K/Pnzc9VVV+WII46odrMAAKqmttoNAADYky1fvrzj9Xe+85188pOfzJNPPtlRNmTIkGo0q1v87ne/y9SpUzN27Nhu/6z169dn4MCB3f45AAC7w4g1AIDd0NjY2HE0NDSkVCp1Kvv2t7+dV7/61Rk0aFBe85rX5Jvf/GbHe/fff/8kybnnnptSqdRx/rvf/S6TJ0/OqFGjMnTo0LzpTW/KnXfeuUvtuueee/LmN785dXV1ecUrXpEJEybk6aef7rh+7bXXZtSoURk2bFje//735+Mf//g2R5/9/ve/T6lUyh//+Me8733vS6lUyvz58/PpT386v/jFLzpG582fP3+r73/ooYdy8sknZ999901DQ0OOP/74/OxnP+tUp1Qq5cYbb8zkyZNTV1eXz372s0mS22+/PUceeWT22muvvOpVr8qnP/3pbNiwoeN9s2bNyqGHHpq6urqMGzcu06ZNS3Nz8y79rgAAihKsAQB0k4ULF+YjH/lILrvssjz++OP54Ac/mL/6q7/K3XffnaQSOCXJvHnzsnz58o7z5ubmnHHGGbnzzjvz85//PKeeemrOOuusPPPMMzv1uRs2bMg555yT448/Po8++mgeeOCB/M3f/E1KpVKS5Lvf/W4+9alP5XOf+1wefvjhjB49OnPmzNnm/caNG5fly5envr4+119/fZYvX57zzz8/l112WQ455JAsX768o2xrXnjhhVx44YVZsmRJHnzwwRx00EE544wz8sILL3Sq96lPfSqTJ0/OY489lve973358Y9/nL/8y7/Mhz/84fz7v/97vva1r2X+/Pn53Oc+1/GeAQMG5Mtf/nIef/zx3HTTTbnrrrtyxRVX7NTvCQBgd5XK5XK52o0AAOgL5s+fnxkzZuT5559PkkyYMCGHHHJI/vEf/7GjznnnnZeWlpb88Ic/TFIZqbVw4cIdbgZwyCGH5EMf+lAuvvjiJJXRbjNmzMiMGTO2qPunP/0p++yzT+65554cf/zxW1w/9thjc/jhh2fu3LkdZUcffXReeumlLFu2bJtteMUrXpHrr7++Y924q666Krfeeut237M1bW1tGT58eL797W/nzDPPTFL5PcyYMSOzZ8/uqDdp0qScfvrpufLKKzvKbr755lxxxRV57rnntnrv733ve/nQhz6U//7v/96lNgEAFGHEGgBAN3niiScyYcKETmUTJkzIE088sd33tbS05IorrsjrX//6vOIVr8jQoUPzq1/9aqdHrI0YMSIXXXRRx0i3G264odNacE888USOOeaYTu/Z/LwrrVq1KlOnTs3BBx+choaGNDQ0pLm5eYvnOeqoozqdP/LII/nMZz6ToUOHdhwf+MAHsnz58rz44otJkrvvvjsnn3xyXvnKV2bYsGF573vfmz/+8Y9paWnptucBANhIsAYA0I02Tr/cqFwub1G2uY9+9KP5/ve/n8997nNZsmRJli1blkMPPTTr1q3b6c+dN29eHnjggRx77LH5zne+k4MPPjgPPvhgoWfYXRdddFEeeeSRXH/99bn//vuzbNmy7LPPPls8T11dXafz9vb2fPrTn86yZcs6jsceeyy/+c1vstdee+Xpp5/OGWeckfHjx+f73/9+HnnkkfzDP/xDksrmBwAA3c2uoAAA3eR1r3tdli5dmve+970dZffff39e97rXdZwPHDgwbW1tnd63ZMmSXHTRRTn33HOTVNZc+/3vf7/Ln/+GN7whb3jDG3LllVfmmGOOybe//e0cffTRed3rXpcHH3ywU7uKhG6DBg3aou1bs2TJksyZMydnnHFGkuTZZ5/dqamab3zjG/Pkk0/mwAMP3Or1hx9+OBs2bMh1112XAQMq/1783e9+dxeeAABg9wjWAAC6yUc/+tGcd955eeMb35gTTzwxt99+e2655ZZOO3zuv//++dd//ddMmDAhgwcPzvDhw3PggQfmlltuyVlnnZVSqZS/+7u/S3t7+05/7lNPPZV//Md/zNlnn50xY8bkySefzK9//euOIO0jH/lILrzwwhx11FGZOHFivvWtb+WXv/xlXvWqV+3S8+2///556qmnsmzZsowdOzbDhg3L4MGDt6h34IEH5pvf/GaOOuqoNDU15aMf/WiGDBmyw/t/8pOfzJlnnplx48blL/7iLzJgwIA8+uijeeyxx/LZz342r371q7Nhw4Z85StfyVlnnZX77rsvN9544y49AwDA7jAVFACgm5xzzjm54YYb8sUvfjGHHHJIvva1r2XevHk54YQTOupcd911Wbx4ccaNG5c3vOENSZLZs2dn+PDhOfbYY3PWWWfl1FNPzRvf+Mad/ty99947v/rVr/KOd7wjBx98cP7mb/4mF198cT74wQ8mSc4///x88pOfzMc+9rEceeSRefrpp/OhD31ol5/vHe94R0477bS89a1vzZ/92Z/ln//5n7da75/+6Z+yevXqvOENb8gFF1yQD3/4wxk5cuQO73/qqafm//7f/5vFixfnTW96U44++ujMmjUr++23X5LkiCOOyKxZs/L5z38+48ePz7e+9a1cc801u/wcAABF2RUUAIDCO3wCAPRnRqwBAAAAQAGCNQAAAAAowFRQAAAAACjAiDUAAAAAKECwBgAAAAAFCNYAAAAAoADBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAAAAAAoQrAEAAABAAf8/A7IuusxObbkAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1500x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def sctter_plot(df,location):\n",
    "    bhk2=df[(df.location==location) & (df.BHK==2)]\n",
    "    bhk3=df[(df.location==location) & (df.BHK==3)]\n",
    "    matplotlib.rcParams[\"figure.figsize\"]=(15,10)\n",
    "    plt.scatter(bhk2.total_sqft,bhk2.price, color=\"blue\", label=\"2 bhk\", s=50)\n",
    "    plt.scatter(bhk3.total_sqft,bhk3.price, marker=\"+\", color=\"red\", label=\"3 bhk\", s=50)\n",
    "    plt.xlabel(\"Total sq ft area\")\n",
    "    plt.ylabel(\"Price\")\n",
    "    plt.title(location)\n",
    "    plt.legend()\n",
    "    \n",
    "sctter_plot(df7, \"Rajaji Nagar\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1c9b4df",
   "metadata": {},
   "source": [
    "From this above plot we can see that all the records where for same location and total_sq_ft area 2bhk priced more than 3bhk are gone now."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "46b37532",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABOIAAANBCAYAAABNqRGIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABK3ElEQVR4nO3de5hV9WHv/88oFwmBHRGZgThBmqhHA9pEPTA89a5cKhJjztGUdKqt1VwEy1GbxFwazGnFpImaExprbSrGaMgflTSNlARjtLWAInGiEuLRBhKMjFgLM0DIgLB+f+TnPhlBIwjf4fJ6Pc9+ntlrfffa3zWTlc3zdu21GqqqqgIAAAAA7FEH9fQEAAAAAOBAIMQBAAAAQAFCHAAAAAAUIMQBAAAAQAFCHAAAAAAUIMQBAAAAQAFCHAAAAAAUIMQBAAAAQAG9enoC+4pt27blueeey4ABA9LQ0NDT0wEAAACgh1RVlfXr12fYsGE56KDXf56bEPc6Pffcc2lubu7paQAAAACwl1i1alWOOOKI1z1eiHudBgwYkOTXv+CBAwf28GwAAAAA6CmdnZ1pbm6u96LXS4h7nV7+OurAgQOFOAAAAAB2+vJlbtYAAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAUIcQAAAABQgBAHAAAAAAX06ukJAK/PkR+/t6ensN9aecO5PT0FAAAADgDOiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAAChAiAMAAACAAoQ4AAAAACigR0PcLbfckuOPPz4DBw7MwIED09LSkn/5l3+pr7/kkkvS0NDQ7TFmzJhu2+jq6sq0adMyePDg9O/fP5MnT86zzz7bbczatWvT2tqaWq2WWq2W1tbWrFu3rsQuAgAAAECSHg5xRxxxRG644YY8+uijefTRR3PmmWfmPe95T5YtW1YfM2HChKxevbr+mDdvXrdtTJ8+PXPnzs2cOXPy0EMPZcOGDZk0aVK2bt1aHzNlypS0tbVl/vz5mT9/ftra2tLa2lpsPwEAAACgV0+++Xnnndft+V/91V/llltuyeLFi/POd74zSdK3b980NTXt8PUdHR356le/mjvvvDNnn312kuTrX/96mpubc99992X8+PFZvnx55s+fn8WLF2f06NFJkttuuy0tLS156qmncswxx+zBPQQAAACAX9trrhG3devWzJkzJxs3bkxLS0t9+QMPPJAhQ4bk6KOPzmWXXZY1a9bU1y1dujRbtmzJuHHj6suGDRuWkSNHZuHChUmSRYsWpVar1SNckowZMya1Wq0+Zke6urrS2dnZ7QEAAAAAu6rHQ9wTTzyRN7/5zenbt28+9KEPZe7cuTnuuOOSJBMnTsxdd92V+++/P1/84hezZMmSnHnmmenq6kqStLe3p0+fPjn00EO7bbOxsTHt7e31MUOGDNnufYcMGVIfsyMzZ86sX1OuVqulubl5d+0yAAAAAAegHv1qapIcc8wxaWtry7p16/KP//iPufjii/Pggw/muOOOy0UXXVQfN3LkyJx00kkZPnx47r333lxwwQWvus2qqtLQ0FB//ps/v9qYV7r22mtz1VVX1Z93dnaKcQAAAADssh4PcX369Mk73vGOJMlJJ52UJUuW5Etf+lJuvfXW7cYOHTo0w4cPz9NPP50kaWpqyubNm7N27dpuZ8WtWbMmY8eOrY95/vnnt9vWCy+8kMbGxledV9++fdO3b983tG8AAAAA8LIe/2rqK1VVVf/q6Su9+OKLWbVqVYYOHZokOfHEE9O7d+8sWLCgPmb16tV58skn6yGupaUlHR0deeSRR+pjHn744XR0dNTHAAAAAMCe1qNnxH3iE5/IxIkT09zcnPXr12fOnDl54IEHMn/+/GzYsCEzZszI+973vgwdOjQrV67MJz7xiQwePDjvfe97kyS1Wi2XXnpprr766hx22GEZNGhQrrnmmowaNap+F9Vjjz02EyZMyGWXXVY/y+7yyy/PpEmT3DEVAAAAgGJ6NMQ9//zzaW1tzerVq1Or1XL88cdn/vz5Oeecc7Jp06Y88cQT+drXvpZ169Zl6NChOeOMM/LNb34zAwYMqG/jpptuSq9evXLhhRdm06ZNOeusszJ79uwcfPDB9TF33XVXrrzyyvrdVSdPnpxZs2YV318AAAAADlwNVVVVPT2JfUFnZ2dqtVo6OjoycODAnp4OB6AjP35vT09hv7XyhnN7egoAAADsQ3a1E+1114gDAAAAgP2REAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFCAEAcAAAAABQhxAAAAAFBAj4a4W265Jccff3wGDhyYgQMHpqWlJf/yL/9SX19VVWbMmJFhw4alX79+Of3007Ns2bJu2+jq6sq0adMyePDg9O/fP5MnT86zzz7bbczatWvT2tqaWq2WWq2W1tbWrFu3rsQuAgAAAECSHg5xRxxxRG644YY8+uijefTRR3PmmWfmPe95Tz22ff7zn8+NN96YWbNmZcmSJWlqaso555yT9evX17cxffr0zJ07N3PmzMlDDz2UDRs2ZNKkSdm6dWt9zJQpU9LW1pb58+dn/vz5aWtrS2tra/H9BQAAAODA1VBVVdXTk/hNgwYNyl//9V/nT/7kTzJs2LBMnz49H/vYx5L8+uy3xsbGfO5zn8sHP/jBdHR05PDDD8+dd96Ziy66KEny3HPPpbm5OfPmzcv48eOzfPnyHHfccVm8eHFGjx6dJFm8eHFaWlryk5/8JMccc8zrmldnZ2dqtVo6OjoycODAPbPz8BqO/Pi9PT2F/dbKG87t6SkAAACwD9nVTrTXXCNu69atmTNnTjZu3JiWlpasWLEi7e3tGTduXH1M3759c9ppp2XhwoVJkqVLl2bLli3dxgwbNiwjR46sj1m0aFFqtVo9wiXJmDFjUqvV6mN2pKurK52dnd0eAAAAALCrejzEPfHEE3nzm9+cvn375kMf+lDmzp2b4447Lu3t7UmSxsbGbuMbGxvr69rb29OnT58ceuihrzlmyJAh273vkCFD6mN2ZObMmfVrytVqtTQ3N7+h/QQAAADgwNbjIe6YY45JW1tbFi9enA9/+MO5+OKL8+Mf/7i+vqGhodv4qqq2W/ZKrxyzo/G/bTvXXnttOjo66o9Vq1a93l0CAAAAgO30eIjr06dP3vGOd+Skk07KzJkzc8IJJ+RLX/pSmpqakmS7s9bWrFlTP0uuqakpmzdvztq1a19zzPPPP7/d+77wwgvbnW33m/r27Vu/m+vLDwAAAADYVT0e4l6pqqp0dXVlxIgRaWpqyoIFC+rrNm/enAcffDBjx45Nkpx44onp3bt3tzGrV6/Ok08+WR/T0tKSjo6OPPLII/UxDz/8cDo6OupjAAAAAGBP69WTb/6JT3wiEydOTHNzc9avX585c+bkgQceyPz589PQ0JDp06fn+uuvz1FHHZWjjjoq119/fd70pjdlypQpSZJarZZLL700V199dQ477LAMGjQo11xzTUaNGpWzzz47SXLsscdmwoQJueyyy3LrrbcmSS6//PJMmjTpdd8xFQAAAADeqB4Ncc8//3xaW1uzevXq1Gq1HH/88Zk/f37OOeecJMlHP/rRbNq0KR/5yEeydu3ajB49Ot/73vcyYMCA+jZuuumm9OrVKxdeeGE2bdqUs846K7Nnz87BBx9cH3PXXXflyiuvrN9ddfLkyZk1a1bZnQUAAADggNZQVVXV05PYF3R2dqZWq6Wjo8P14ugRR3783p6ewn5r5Q3n9vQUAAAA2Ifsaifa664RBwAAAAD7IyEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACgACEOAAAAAAoQ4gAAAACggB4NcTNnzszJJ5+cAQMGZMiQITn//PPz1FNPdRtzySWXpKGhodtjzJgx3cZ0dXVl2rRpGTx4cPr375/Jkyfn2Wef7TZm7dq1aW1tTa1WS61WS2tra9atW7endxEAAAAAkvRwiHvwwQdzxRVXZPHixVmwYEFeeumljBs3Lhs3buw2bsKECVm9enX9MW/evG7rp0+fnrlz52bOnDl56KGHsmHDhkyaNClbt26tj5kyZUra2toyf/78zJ8/P21tbWltbS2ynwAAAADQqyfffP78+d2e33777RkyZEiWLl2aU089tb68b9++aWpq2uE2Ojo68tWvfjV33nlnzj777CTJ17/+9TQ3N+e+++7L+PHjs3z58syfPz+LFy/O6NGjkyS33XZbWlpa8tRTT+WYY47ZQ3sIAAAAAL+2V10jrqOjI0kyaNCgbssfeOCBDBkyJEcffXQuu+yyrFmzpr5u6dKl2bJlS8aNG1dfNmzYsIwcOTILFy5MkixatCi1Wq0e4ZJkzJgxqdVq9TGv1NXVlc7Ozm4PAAAAANhVe02Iq6oqV111VX7v934vI0eOrC+fOHFi7rrrrtx///354he/mCVLluTMM89MV1dXkqS9vT19+vTJoYce2m17jY2NaW9vr48ZMmTIdu85ZMiQ+phXmjlzZv16crVaLc3NzbtrVwEAAAA4APXoV1N/09SpU/P444/noYce6rb8oosuqv88cuTInHTSSRk+fHjuvffeXHDBBa+6vaqq0tDQUH/+mz+/2pjfdO211+aqq66qP+/s7BTjAAAAANhle8UZcdOmTcu3v/3t/OAHP8gRRxzxmmOHDh2a4cOH5+mnn06SNDU1ZfPmzVm7dm23cWvWrEljY2N9zPPPP7/dtl544YX6mFfq27dvBg4c2O0BAAAAALuqR0NcVVWZOnVq7rnnntx///0ZMWLEb33Niy++mFWrVmXo0KFJkhNPPDG9e/fOggUL6mNWr16dJ598MmPHjk2StLS0pKOjI4888kh9zMMPP5yOjo76GAAAAADYk3r0q6lXXHFF7r777vzTP/1TBgwYUL9eW61WS79+/bJhw4bMmDEj73vf+zJ06NCsXLkyn/jEJzJ48OC8973vrY+99NJLc/XVV+ewww7LoEGDcs0112TUqFH1u6gee+yxmTBhQi677LLceuutSZLLL788kyZNcsdUAAAAAIro0RB3yy23JElOP/30bstvv/32XHLJJTn44IPzxBNP5Gtf+1rWrVuXoUOH5owzzsg3v/nNDBgwoD7+pptuSq9evXLhhRdm06ZNOeusszJ79uwcfPDB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6vrjl7/8ZX2Mz1DoOb/tGN0XP0OFOGCXXXTRRdXQoUOr3r17V8OGDasuuOCCatmyZfX127Ztqz7zmc9UTU1NVd++fatTTz21euKJJ7ptY9OmTdXUqVOrQYMGVf369asmTZpU/fznP+825sUXX6w+8IEPVAMGDKgGDBhQfeADH6jWrl1bYhdhn/GDH/ygSrLd4+KLL66qquzx+LOf/aw699xzq379+lWDBg2qpk6dWv3qV7/ak7sPe73XOkZ/+ctfVuPGjasOP/zwqnfv3tXb3va26uKLL97u+HOMwu63o+MySXX77bfXx/gMhZ7z247RffEztOH/3zEAAAAAYA9yjTgAAAAAKECIAwAAAIAChDgAAAAAKECIAwAAAIAChDgAAAAAKECIAwAAAIAChDgAAAAAKECIAwDYTVauXJmGhoa0tbX19FR4FX/3d3+X5ubmHHTQQbn55pt7ejoAwAGmoaqqqqcnAQCwP9i6dWteeOGFDB48OL169erp6fAKnZ2dGTx4cG688ca8733vS61Wy5ve9KY3vN0ZM2bkW9/6lgALAPxW/oUIALAbbN68OX369ElTU1NPT2Wnbd26NQ0NDTnooH3jyxIv/6531s9//vNs2bIl5557boYOHboHZgYA8Nr2jX9tAQAUdPrpp2fq1KmZOnVq3vKWt+Swww7Lpz71qfzmFwmOPPLI/OVf/mUuueSS1Gq1XHbZZTv8auqyZcty7rnnZuDAgRkwYEBOOeWU/Md//Ed9/e23355jjz02hxxySP7bf/tv+cpXvvKG57Z58+Z89KMfzVvf+tb0798/o0ePzgMPPFBfP3v27LzlLW/Jd77znRx33HHp27dvfvazn233XmvXrs0HPvCBHH744enXr1+OOuqo3H777fX1jzzySN71rnflkEMOyUknnZS5c+d22/+X3+c3fetb30pDQ0P9+X/8x3/kPe95TxobG/PmN785J598cu67775ur9nR7zpJFi5cmFNPPTX9+vVLc3NzrrzyymzcuHGHv7fZs2dn1KhRSZLf+Z3fSUNDQ1auXJkk+ed//ueceOKJOeSQQ/I7v/M7ue666/LSSy/VX9vR0ZHLL788Q4YMycCBA3PmmWfmRz/6UX271113XX70ox+loaEhDQ0NmT179g7nAAAgxAEA7MAdd9yRXr165eGHH87/+T//JzfddFP+/u//vtuYv/7rv87IkSOzdOnSfPrTn95uG7/4xS9y6qmn5pBDDsn999+fpUuX5k/+5E/qkee2227LJz/5yfzVX/1Vli9fnuuvvz6f/vSnc8cdd7yhuf3xH/9x/v3f/z1z5szJ448/nv/5P/9nJkyYkKeffro+5pe//GVmzpyZv//7v8+yZcsyZMiQ7d7n05/+dH784x/nX/7lX7J8+fLccsstGTx4cJJk48aNmTRpUo455pgsXbo0M2bMyDXXXPP6f8H/vw0bNuT3f//3c9999+Wxxx7L+PHjc9555+XnP/95t3Gv/F0/8cQTGT9+fC644II8/vjj+eY3v5mHHnooU6dO3eH7XHTRRfXA98gjj2T16tVpbm7Od7/73fzhH/5hrrzyyvz4xz/OrbfemtmzZ+ev/uqvkiRVVeXcc89Ne3t75s2bl6VLl+bd7353zjrrrPzXf/1XLrroolx99dV55zvfmdWrV2f16tW56KKLdvr3AAAcICoAALo57bTTqmOPPbbatm1bfdnHPvax6thjj60/Hz58eHX++ed3e92KFSuqJNVjjz1WVVVVXXvttdWIESOqzZs37/B9mpubq7vvvrvbsv/9v/931dLSsstze+aZZ6qGhobqF7/4RbfXnXXWWdW1115bVVVV3X777VWSqq2t7VXfp6qq6rzzzqv++I//eIfrbr311mrQoEHVxo0b68tuueWWbvt/++23V7Vardvr5s6dW/22f4Ied9xx1Ze//OX68x39rltbW6vLL7+827J/+7d/qw466KBq06ZNO9zuY489ViWpVqxYUV92yimnVNdff323cXfeeWc1dOjQqqqq6vvf/341cODA6le/+lW3MW9/+9urW2+9taqqqvrMZz5TnXDCCa+5TwAAVVVVrhEHALADY8aM6fYVypaWlnzxi1/M1q1bc/DBBydJTjrppNfcRltbW0455ZT07t17u3UvvPBCVq1alUsvvbT+Vcskeemll1Kr1XZ5bj/84Q9TVVWOPvrobq/p6urKYYcdVn/ep0+fHH/88a/5Ph/+8Ifzvve9Lz/84Q8zbty4nH/++Rk7dmySZPny5TnhhBO63eygpaXlNbe3Ixs3bsx1112X73znO3nuuefy0ksvZdOmTdudEffK3/XSpUvzzDPP5K677qovq6oq27Zty4oVK3Lssce+rvdfunRplixZUj8DLvn1NfN+9atf5Ze//GWWLl2aDRs2dPvdJcmmTZu6fcUYAOD1EOIAAHZR//79X3N9v379XnXdtm3bkvz666mjR4/utu7l0Lcrtm3bloMPPjhLly7dbjtvfvObu83tN2PejkycODE/+9nPcu+99+a+++7LWWedlSuuuCJf+MIXul2T7tUcdNBB243bsmVLt+d//ud/nu9+97v5whe+kHe84x3p169f/sf/+B/ZvHlzt3Gv/F1v27YtH/zgB3PllVdu975ve9vbfuvcfnM71113XS644ILt1h1yyCHZtm1bhg4d2u0aey975fXvAAB+GyEOAGAHFi9evN3zo446aqci2fHHH5877rgjW7Zs2e6suMbGxrz1rW/NT3/603zgAx/YbXN717vela1bt2bNmjU55ZRTdmq7O3L44YfnkksuySWXXJJTTjklf/7nf54vfOELOe6443LnnXdm06ZN9eD4ynkdfvjhWb9+fTZu3FgPab95I4sk+bd/+7dccsklee9735vk19eMe/kmCq/l3e9+d5YtW5Z3vOMdb2j/3v3ud+epp5561e28+93vTnt7e3r16pUjjzxyh2P69OmTrVu3vqF5AAAHBjdrAADYgVWrVuWqq67KU089lW984xv58pe/nD/7sz/bqW1MnTo1nZ2def/7359HH300Tz/9dO6888489dRTSZIZM2Zk5syZ+dKXvpT/+3//b5544oncfvvtufHGG3d5bkcffXQ+8IEP5I/+6I9yzz33ZMWKFVmyZEk+97nPZd68eTs1/7/4i7/IP/3TP+WZZ57JsmXL8p3vfKf+lc8pU6bkoIMOyqWXXpof//jHmTdvXr7whS90e/3o0aPzpje9KZ/4xCfyzDPP5O67797ujqLveMc7cs8996StrS0/+tGPMmXKlPrZgq/lYx/7WBYtWpQrrrgibW1tefrpp/Ptb38706ZN2+l9/NrXvpYZM2Zk2bJlWb58eb75zW/mU5/6VJLk7LPPTktLS84///x897vfzcqVK7Nw4cJ86lOfyqOPPprk13d1XbFiRdra2vKf//mf6erq2qk5AAAHDiEOAGAH/uiP/iibNm3Kf//v/z1XXHFFpk2blssvv3yntnHYYYfl/vvvz4YNG3LaaaflxBNPzG233VY/O+5P//RP8/d///eZPXt2Ro0aldNOOy2zZ8/OiBEj3tDcbr/99vzRH/1Rrr766hxzzDGZPHlyHn744TQ3N+/U/Pv06ZNrr702xx9/fE499dQcfPDBmTNnTpJff831n//5n/PjH/8473rXu/LJT34yn/vc57q9ftCgQfn617+eefPmZdSoUfnGN76RGTNmdBtz00035dBDD83YsWNz3nnnZfz48Xn3u9/9W+d2/PHH58EHH8zTTz+dU045Je9617vy6U9/OkOHDt2pfRw/fny+853vZMGCBTn55JMzZsyY3HjjjRk+fHiSpKGhIfPmzcupp56aP/mTP8nRRx+d97///Vm5cmUaGxuTJO973/syYcKEnHHGGTn88MPzjW98Y6fmAAAcOBqq13OBDwCAA8jpp5+e3/3d383NN9/c01PZzt48t5UrV2bEiBF57LHH8ru/+7s9PR0AgL2OM+IAAAAAoAAhDgAAAAAK8NVUAAAAACjAGXEAAAAAUIAQBwAAAAAFCHEAAAAAUIAQBwAAAAAFCHEAAAAAUIAQBwAAAAAFCHEAAAAAUIAQBwAAAAAFCHEAAAAAUMD/B0c5ZJoZimsDAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1500x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#to check which value of price_per_sqft feature is the most common in the dataset.\n",
    "plt.hist(df7.price_per_sqft, rwidth=0.8)\n",
    "plt.xlabel(\"price per square feet\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a31635b",
   "metadata": {},
   "source": [
    "This histogram plot shows that most of the properties lie between 0-10k price range."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abc5a566",
   "metadata": {},
   "source": [
    "### Outlier removal using bath feature:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "0c0c60a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4.,  3.,  2.,  5.,  8.,  1.,  6.,  7.,  9., 12., 16., 13.])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df7[\"bath\"].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73de9b70",
   "metadata": {},
   "source": [
    "Here we can see that few properties are listed with even more than 12 bathrooms, which seems unusual and we need remove those records which have 2 more bathrooms than the number of rooms in a home."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "88bd8755",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Count')"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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BAAAAKCQzelENfvMNwAAAADY0YlvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYA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AQAAANi51FUqlUqth3jB008/nd69e2f27Nk5+uijU6lU0tTUlJaWllxyySVJ/naVW0NDQ6ZOnZqzzjorra2t2WuvvXLzzTfnlFNOSZI8+eST6dOnT773ve9l9OjRWbRoUQYNGpS5c+dm6NChSZK5c+dm2LBheeSRRzJgwIBXnW3VqlWpr69Pa2trevToUe4P4XXW79Lv1nqEHdJjU06o9QgAAABAIZvTibarz3xrbW1NkvTs2TNJsmTJkixfvjyjRo2qrunSpUtGjBiROXPmJEkWLFiQ5557rs2apqamDB48uLrmvvvuS319fTW8JcmRRx6Z+vr66poXW7duXVatWtXmBgAAAACbY7uJb5VKJRdccEGOOuqoDB48OEmyfPnyJElDQ0ObtQ0NDdV9y5cvT+fOnbPHHnu84prevXtv8pq9e/eurnmxyZMnVz8frr6+Pn369Nm6AwQAAABgp7PdxLdzzz03v/zlL/P1r399k311dXVt7lcqlU22vdiL17zU+ld6nssuuyytra3V29KlS1/LYQAAAABA1XYR38aPH5877rgjd999d/bZZ5/q9sbGxiTZ5Oq0FStWVK+Ga2xszPr167Ny5cpXXPPUU09t8rpPP/30JlfVvaBLly7p0aNHmxsAAAAAbI6axrdKpZJzzz033/zmN/OjH/0o++23X5v9++23XxobGzNr1qzqtvXr12f27NkZPnx4kuSwww5Lp06d2qxZtmxZHnrooeqaYcOGpbW1NfPmzauuuf/++9Pa2lpdAwAAAADbWsdavvg555yTW2+9Nd/61rfSvXv36hVu9fX16dq1a+rq6tLS0pJJkyalf//+6d+/fyZNmpRdd90148aNq649/fTTM2HChPTq1Ss9e/bMhRdemCFDhmTkyJFJkoEDB+b444/PGWeckRtuuCFJcuaZZ2bMmDGv6ZtOAQAAAGBL1DS+XX/99UmSY445ps32r3zlK/nABz6QJLn44ouzdu3anH322Vm5cmWGDh2aO++8M927d6+uv/rqq9OxY8ecfPLJWbt2bY499tjcdNNN6dChQ3XN1772tZx33nnVb0UdO3Zspk+fXvYAAQAAANip1VUqlUqth2gPVq1alfr6+rS2tu5Qn//W79Lv1nqEHdJjU06o9QgAAABAIZvTibaLL1wAAAAAgB2R+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFCI+AYAAAAAhYhvAAAAAFBIx1oPALw2/S79bq1H2GE9NuWEWo8AAADADsqVbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIWIbwAAAABQiPgGAAAAAIXUNL79+Mc/zjvf+c40NTWlrq4ut99+e5v9lUolEydOTFNTU7p27ZpjjjkmDz/8cJs169aty/jx47PnnnumW7duGTt2bJ544ok2a1auXJnm5ubU19envr4+zc3NeeaZZwofHQAAAAA7u5rGt2effTYHH3xwpk+f/pL7r7rqqkybNi3Tp0/P/Pnz09jYmOOOOy6rV6+urmlpacmMGTNy22235d57782aNWsyZsyYbNiwobpm3LhxWbhwYWbOnJmZM2dm4cKFaW5uLn58AAAAAOzcOtbyxd/+9rfn7W9/+0vuq1Qqueaaa/LRj34073rXu5IkX/3qV9PQ0JBbb701Z511VlpbW/OlL30pN998c0aOHJkkueWWW9KnT5/88Ic/zOjRo7No0aLMnDkzc+fOzdChQ5MkX/ziFzNs2LAsXrw4AwYMeH0OFgAAAICdznb7mW9LlizJ8uXLM2rUqOq2Ll26ZMSIEZkzZ06SZMGCBXnuuefarGlqasrgwYOra+67777U19dXw1uSHHnkkamvr6+uAQAAAIASanrl2ytZvnx5kqShoaHN9oaGhvz+97+vruncuXP22GOPTda88Pjly5end+/emzx/7969q2teyrp167Ju3brq/VWrVm3ZgQAAAACw09pur3x7QV1dXZv7lUplk20v9uI1L7X+1Z5n8uTJ1S9oqK+vT58+fTZzcgAAAAB2dtttfGtsbEySTa5OW7FiRfVquMbGxqxfvz4rV658xTVPPfXUJs//9NNPb3JV3d+77LLL0traWr0tXbp0q44HAAAAgJ3Pdhvf9ttvvzQ2NmbWrFnVbevXr8/s2bMzfPjwJMlhhx2WTp06tVmzbNmyPPTQQ9U1w4YNS2tra+bNm1ddc//996e1tbW65qV06dIlPXr0aHMDAAAAgM1R0898W7NmTX7zm99U7y9ZsiQLFy5Mz54907dv37S0tGTSpEnp379/+vfvn0mTJmXXXXfNuHHjkiT19fU5/fTTM2HChPTq1Ss9e/bMhRdemCFDhlS//XTgwIE5/vjjc8YZZ+SGG25Ikpx55pkZM2aMbzoFAAAAoKiaxref/exnedvb3la9f8EFFyRJTj311Nx00025+OKLs3bt2px99tlZuXJlhg4dmjvvvDPdu3evPubqq69Ox44dc/LJJ2ft2rU59thjc9NNN6VDhw7VNV/72tdy3nnnVb8VdezYsZk+ffrrdJQAAAAA7KzqKpVKpdZDtAerVq1KfX19Wltbd6hfQe136XdrPcIO6bEpJ2zz5/RelVPi/QIAAGDHtTmdaLv9zDcAAAAAaO/ENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgELENwAAAAAoRHwDAAAAgEI61noAgB1Rv0u/W+sRdliPTTmh1iMAAAC8Zq58AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCxDcAAAAAKER8AwAAAIBCOtZ6AACotX6XfrfWI+yQHptyQq1HAACAmnPlGwAAAAAUIr4BAAAAQCHiGwAAAAAUIr4BAAAAQCHiGwAAAAAUIr4BAAAAQCHiGwAAAAAU0rHWA7yePv/5z+fTn/50li1blgMPPDDXXHNN3vrWt9Z6LABgM/S79Lu1HmGH9NiUE2o9AgDADmmnufLtG9/4RlpaWvLRj340P//5z/PWt741b3/72/P444/XejQAAAAAdlA7TXybNm1aTj/99Pz7v/97Bg4cmGuuuSZ9+vTJ9ddfX+vRAAAAANhB7RTxbf369VmwYEFGjRrVZvuoUaMyZ86cGk0FAAAAwI5up/jMtz/+8Y/ZsGFDGhoa2mxvaGjI8uXLX/Ix69aty7p166r3W1tbkySrVq0qN2gNbFz3l1qPsEMqcZ54r8rxfrUv3q/2o9R/M71fZZR6vwZf/oMiz7uze+iK0bUegRrz71Y5/v0CXosXfnaqVCqvunaniG8vqKura3O/Uqlssu0FkydPzhVXXLHJ9j59+hSZjR1L/TW1noDN4f1qX7xf7Yf3qn3xfrUv3i8ox79fwOZYvXp16uvrX3HNThHf9txzz3To0GGTq9xWrFixydVwL7jssstywQUXVO9v3Lgxf/7zn9OrV6+XDXZs/1atWpU+ffpk6dKl6dGjR63HoZ1x/rClnDtsDecPW8P5w9Zw/rClnDtsjfZy/lQqlaxevTpNTU2vunaniG+dO3fOYYcdllmzZuVf/uVfqttnzZqVE0888SUf06VLl3Tp0qXNtt13373kmLyOevTosV3/S8z2zfnDlnLusDWcP2wN5w9bw/nDlnLusDXaw/nzale8vWCniG9JcsEFF6S5uTmHH354hg0blhtvvDGPP/54PvShD9V6NAAAAAB2UDtNfDvllFPypz/9KVdeeWWWLVuWwYMH53vf+1723XffWo8GAAAAwA5qp4lvSXL22Wfn7LPPrvUY1FCXLl1y+eWXb/IrxfBaOH/YUs4dtobzh63h/GFrOH/YUs4dtsaOeP7UVV7Ld6ICAAAAAJvtDbUeAAAAAAB2VOIbAAAAABQivgEAAABAIeIbAAAAABQivrHDmzx5co444oh07949vXv3zkknnZTFixfXeizaqcmTJ6euri4tLS21HoV24g9/+EP+7d/+Lb169cquu+6aQw45JAsWLKj1WLQDzz//fD72sY9lv/32S9euXbP//vvnyiuvzMaNG2s9GtuhH//4x3nnO9+Zpqam1NXV5fbbb2+zv1KpZOLEiWlqakrXrl1zzDHH5OGHH67NsGxXXuncee6553LJJZdkyJAh6datW5qamvL+978/Tz75ZO0GZrvyan/3/L2zzjordXV1ueaaa163+di+vZbzZ9GiRRk7dmzq6+vTvXv3HHnkkXn88cdf/2G3kvjGDm/27Nk555xzMnfu3MyaNSvPP/98Ro0alWeffbbWo9HOzJ8/PzfeeGMOOuigWo9CO7Fy5cq85S1vSadOnfL9738/v/rVr/KZz3wmu+++e61Hox2YOnVqvvCFL2T69OlZtGhRrrrqqnz605/OtddeW+vR2A49++yzOfjggzN9+vSX3H/VVVdl2rRpmT59eubPn5/GxsYcd9xxWb169es8KdubVzp3/vKXv+SBBx7Ixz/+8TzwwAP55je/mV//+tcZO3ZsDSZle/Rqf/e84Pbbb8/999+fpqam12ky2oNXO39++9vf5qijjsoBBxyQe+65J7/4xS/y8Y9/PLvsssvrPOnWq6tUKpVaDwGvp6effjq9e/fO7Nmzc/TRR9d6HNqJNWvW5B//8R/z+c9/Pp/85CdzyCGH+L92vKpLL700P/3pT/OTn/yk1qPQDo0ZMyYNDQ350pe+VN32r//6r9l1111z880313Aytnd1dXWZMWNGTjrppCR/u+qtqakpLS0tueSSS5Ik69atS0NDQ6ZOnZqzzjqrhtOyPXnxufNS5s+fn3/6p3/K73//+/Tt2/f1G47t3sudP3/4wx8ydOjQ/OAHP8gJJ5yQlpYWv0XCJl7q/HnPe96TTp067RA/97jyjZ1Oa2trkqRnz541noT25JxzzskJJ5yQkSNH1noU2pE77rgjhx9+eN797nend+/eOfTQQ/PFL36x1mPRThx11FG566678utf/zpJ8otf/CL33ntv3vGOd9R4MtqbJUuWZPny5Rk1alR1W5cuXTJixIjMmTOnhpPRHrW2tqaurs5V3LwmGzduTHNzcy666KIceOCBtR6HdmTjxo357ne/mze/+c0ZPXp0evfunaFDh77irzZvz8Q3diqVSiUXXHBBjjrqqAwePLjW49BO3HbbbXnggQcyefLkWo9CO/O73/0u119/ffr3758f/OAH+dCHPpTzzjsv//3f/13r0WgHLrnkkrz3ve/NAQcckE6dOuXQQw9NS0tL3vve99Z6NNqZ5cuXJ0kaGhrabG9oaKjug9fir3/9ay699NKMGzcuPXr0qPU4tANTp05Nx44dc95559V6FNqZFStWZM2aNZkyZUqOP/743HnnnfmXf/mXvOtd78rs2bNrPd5m61jrAeD1dO655+aXv/xl7r333lqPQjuxdOnSnH/++bnzzjvb5WcLUFsbN27M4YcfnkmTJiVJDj300Dz88MO5/vrr8/73v7/G07G9+8Y3vpFbbrklt956aw488MAsXLgwLS0taWpqyqmnnlrr8WiH6urq2tyvVCqbbIOX89xzz+U973lPNm7cmM9//vO1Hod2YMGCBfnsZz+bBx54wN81bLYXvmDqxBNPzEc+8pEkySGHHJI5c+bkC1/4QkaMGFHL8TabK9/YaYwfPz533HFH7r777uyzzz61Hod2YsGCBVmxYkUOO+ywdOzYMR07dszs2bPzuc99Lh07dsyGDRtqPSLbsb333juDBg1qs23gwIHt8huaeP1ddNFFufTSS/Oe97wnQ4YMSXNzcz7ykY+4CpfN1tjYmCSbXOW2YsWKTa6Gg5fy3HPP5eSTT86SJUsya9YsV73xmvzkJz/JihUr0rdv3+rP0b///e8zYcKE9OvXr9bjsZ3bc88907Fjxx3mZ2lXvrHDq1QqGT9+fGbMmJF77rkn++23X61Hoh059thj8+CDD7bZ9sEPfjAHHHBALrnkknTo0KFGk9EevOUtb8nixYvbbPv1r3+dfffdt0YT0Z785S9/yRve0Pb/k3bo0KH6f4Lhtdpvv/3S2NiYWbNm5dBDD02SrF+/PrNnz87UqVNrPB3buxfC26OPPpq77747vXr1qvVItBPNzc2bfF7y6NGj09zcnA9+8IM1mor2onPnzjniiCN2mJ+lxTd2eOecc05uvfXWfOtb30r37t2r/9e3vr4+Xbt2rfF0bO+6d+++yecDduvWLb169fK5gbyqj3zkIxk+fHgmTZqUk08+OfPmzcuNN96YG2+8sdaj0Q68853vzKc+9an07ds3Bx54YH7+859n2rRpOe2002o9GtuhNWvW5De/+U31/pIlS7Jw4cL07Nkzffv2TUtLSyZNmpT+/funf//+mTRpUnbdddeMGzeuhlOzPXilc6epqSn/63/9rzzwwAP5zne+kw0bNlR/lu7Zs2c6d+5cq7HZTrza3z0vjrWdOnVKY2NjBgwY8HqPynbo1c6fiy66KKecckqOPvrovO1tb8vMmTPz7W9/O/fcc0/tht5SFdjBJXnJ21e+8pVaj0Y7NWLEiMr5559f6zFoJ7797W9XBg8eXOnSpUvlgAMOqNx44421Hol2YtWqVZXzzz+/0rdv38ouu+xS2X///Ssf/ehHK+vWrav1aGyH7r777pf8eefUU0+tVCqVysaNGyuXX355pbGxsdKlS5fK0UcfXXnwwQdrOzTbhVc6d5YsWfKyP0vffffdtR6d7cCr/d3zYvvuu2/l6quvfl1nZPv1Ws6fL33pS5U3velNlV122aVy8MEHV26//fbaDbwV6iqVSqV84gMAAACAnY8vXAAAAACAQsQ3AAAAAChEfAMAAACAQsQ3AAAAAChEfAMAAACAQsQ3AAAAAChEfAMAAACAQsQ3AIDtwGOPPZa6urosXLiw1qNUPfLIIznyyCOzyy675JBDDnnJNcccc0xaWlqKvH7J5wYAeL2IbwAAST7wgQ+krq4uU6ZMabP99ttvT11dXY2mqq3LL7883bp1y+LFi3PXXXcVe5177rkndXV1eeaZZ4q9BgBArYhvAAD/Y5dddsnUqVOzcuXKWo+yzaxfv36LH/vb3/42Rx11VPbdd9/06tVrG05VznPPPVfrEQAA2hDfAAD+x8iRI9PY2JjJkye/7JqJEydu8iuY11xzTfr161e9/4EPfCAnnXRSJk2alIaGhuy+++654oor8vzzz+eiiy5Kz549s88+++TLX/7yJs//yCOPZPjw4dlll11y4IEH5p577mmz/1e/+lXe8Y53ZLfddktDQ0Oam5vzxz/+sbr/mGOOybnnnpsLLrgge+65Z4477riXPI6NGzfmyiuvzD777JMuXbrkkEMOycyZM6v76+rqsmDBglx55ZWpq6vLxIkTX/bP5Pnnn8+5556b3XffPb169crHPvaxVCqV6v5bbrklhx9+eLp3757GxsaMGzcuK1asSPK3X7d929veliTZY489UldXlw984ANt5rz44ovTs2fPNDY2bjJHXV1dvvCFL+TEE09Mt27d8slPfjJJcv311+eNb3xjOnfunAEDBuTmm29u87jHH388J554Ynbbbbf06NEjJ598cp566qnq/hfe5y9/+cvp27dvdtttt3z4wx/Ohg0bctVVV6WxsTG9e/fOpz71qTbPO3HixPTt2zddunRJU1NTzjvvvJf9cwMAdg7iGwDA/+jQoUMmTZqUa6+9Nk888cRWPdePfvSjPPnkk/nxj3+cadOmZeLEiRkzZkz22GOP3H///fnQhz6UD33oQ1m6dGmbx1100UWZMGFCfv7zn2f48OEZO3Zs/vSnPyVJli1blhEjRuSQQw7Jz372s8ycOTNPPfVUTj755DbP8dWvfjUdO3bMT3/609xwww0vOd9nP/vZfOYzn8l//ud/5pe//GVGjx6dsWPH5tFHH62+1oEHHpgJEyZk2bJlufDCC1/2WF94vfvvvz+f+9zncvXVV+e//uu/qvvXr1+fT3ziE/nFL36R22+/PUuWLKkGtj59+uT//b//lyRZvHhxli1bls9+9rNtnrtbt265//77c9VVV+XKK6/MrFmz2rz+5ZdfnhNPPDEPPvhgTjvttMyYMSPnn39+JkyYkIceeihnnXVWPvjBD+buu+9OklQqlZx00kn585//nNmzZ2fWrFn57W9/m1NOOaXN8/72t7/N97///cycOTNf//rX8+UvfzknnHBCnnjiicyePTtTp07Nxz72scydOzdJ8n//7//N1VdfnRtuuCGPPvpobr/99gwZMuRl/9wAgJ1EBQCAyqmnnlo58cQTK5VKpXLkkUdWTjvttEqlUqnMmDGj8vc/Ml1++eWVgw8+uM1jr7766sq+++7b5rn23XffyoYNG6rbBgwYUHnrW99avf/8889XunXrVvn6179eqVQqlSVLllSSVKZMmVJd89xzz1X22WefytSpUyuVSqXy8Y9/vDJq1Kg2r7106dJKksrixYsrlUqlMmLEiMohhxzyqsfb1NRU+dSnPtVm2xFHHFE5++yzq/cPPvjgyuWXX/6KzzNixIjKwIEDKxs3bqxuu+SSSyoDBw582cfMmzevkqSyevXqSqVSqdx9992VJJWVK1du8txHHXXUJjNecskl1ftJKi0tLW3WDB8+vHLGGWe02fbud7+78o53vKNSqVQqd955Z6VDhw6Vxx9/vLr/4YcfriSpzJs3r1Kp/O193nXXXSurVq2qrhk9enSlX79+m7yvkydPrlQqlcpnPvOZypvf/ObK+vXrX/bYAYCdjyvfAABeZOrUqfnqV7+aX/3qV1v8HAceeGDe8Ib//0ethoaGNldBdejQIb169ar++uULhg0bVv3njh075vDDD8+iRYuSJAsWLMjdd9+d3XbbrXo74IADkvztKq0XHH744a8426pVq/Lkk0/mLW95S5vtb3nLW6qvtTmOPPLINl9KMWzYsDz66KPZsGFDkuTnP/95TjzxxOy7777p3r17jjnmmCR/+9XPV3PQQQe1ub/33ntv8mf24uNdtGjRKx7bokWL0qdPn/Tp06e6f9CgQdl9993bHH+/fv3SvXv36v2GhoYMGjRok/f1hXne/e53Z+3atdl///1zxhlnZMaMGXn++edf9RgBgB2b+AYA8CJHH310Ro8enf/4j//YZN8b3vCGNp9nlrz0h/x36tSpzf26urqX3LZx48ZXneeFsLVx48a8853vzMKFC9vcHn300Rx99NHV9d26dXvV5/z7531BpVLZ5t/s+uyzz2bUqFHZbbfdcsstt2T+/PmZMWNGktf2ZRCv5c/spY73lY7t5Y7zxds39z3s06dPFi9enOuuuy5du3bN2WefnaOPPtqXQADATk58AwB4CVOmTMm3v/3tzJkzp832vfbaK8uXL28T4BYuXLjNXveFzw9L/vZFBgsWLKhe3faP//iPefjhh9OvX7+86U1vanN7rcEtSXr06JGmpqbce++9bbbPmTMnAwcO3KqZX7jfv3//dOjQIY888kj++Mc/ZsqUKXnrW9+aAw44YJMr1zp37pwk1SvlttbAgQNf8dgGDRqUxx9/vM3n7f3qV79Ka2vrFh3/3+vatWvGjh2bz33uc7nnnnty33335cEHH9yq5wQA2jfxDQDgJQwZMiTve9/7cu2117bZfswxx+Tpp5/OVVddld/+9re57rrr8v3vf3+bve51112XGTNm5JFHHsk555yTlStX5rTTTkuSnHPOOfnzn/+c9773vZk3b15+97vf5c4778xpp5222eHqoosuytSpU/ONb3wjixcvzqWXXpqFCxfm/PPP3+yZly5dmgsuuCCLFy/O17/+9Vx77bXV5+nbt286d+6ca6+9Nr/73e9yxx135BOf+ESbx++7776pq6vLd77znTz99NNZs2bNZs/w4mO76aab8oUvfCGPPvpopk2blm9+85vVL40YOXJkDjrooLzvfe/LAw88kHnz5uX9739/RowY8aq/svtKbrrppnzpS1/KQw89lN/97ne5+eab07Vr1+y7775bdTwAQPsmvgEAvIxPfOITm/yK6cCBA/P5z38+1113XQ4++ODMmzfvFb8JdHNNmTIlU6dOzcEHH5yf/OQn+da3vpU999wzSdLU1JSf/vSn2bBhQ0aPHp3Bgwfn/PPPT319fZvPIXstzjvvvEyYMCETJkzIkCFDMnPmzNxxxx3p37//Zs/8/ve/P2vXrs0//dM/5Zxzzsn48eNz5plnJvnblYI33XRT/s//+T8ZNGhQpkyZkv/8z/9s8/h/+Id/yBVXXJFLL700DQ0NOffcczd7hr930kkn5bOf/Ww+/elP58ADD8wNN9yQr3zlK9XPmqurq8vtt9+ePfbYI0cffXRGjhyZ/fffP9/4xje26nV33333fPGLX8xb3vKWHHTQQbnrrrvy7W9/O7169dqq5wUA2re6yot/ogQAAAAAtglXvgEAAABAIeIbAAAAABQivgEAAABAIeIbAAAAABQivgEAAABAIeIbAAAAABQivgEAAABAIeIbAAAAABQivgEAAABAIeIbAAAAABQivgEAAABAIeIbAAAAABTy/wGa84OHM7w+lQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1500x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df7.bath, rwidth=0.8)\n",
    "plt.xlabel(\"Number of bathrooms\")\n",
    "plt.ylabel(\"Count\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cdb95428",
   "metadata": {},
   "source": [
    "So most of the properties have bathrooms in range 2-6. But few outliers can be seen in the plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "6c5660e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1626</th>\n",
       "      <td>Chikkabanavar</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2460.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>4</td>\n",
       "      <td>3252.032520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5238</th>\n",
       "      <td>Nagasandra</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>7000.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>450.0</td>\n",
       "      <td>4</td>\n",
       "      <td>6428.571429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6711</th>\n",
       "      <td>Thanisandra</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1806.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>116.0</td>\n",
       "      <td>3</td>\n",
       "      <td>6423.034330</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8411</th>\n",
       "      <td>other</td>\n",
       "      <td>6 BHK</td>\n",
       "      <td>11338.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>6</td>\n",
       "      <td>8819.897689</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           location       size  total_sqft  bath   price  BHK  price_per_sqft\n",
       "1626  Chikkabanavar  4 Bedroom      2460.0   7.0    80.0    4     3252.032520\n",
       "5238     Nagasandra  4 Bedroom      7000.0   8.0   450.0    4     6428.571429\n",
       "6711    Thanisandra      3 BHK      1806.0   6.0   116.0    3     6423.034330\n",
       "8411          other      6 BHK     11338.0   9.0  1000.0    6     8819.897689"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for all the properties which has 2 more bathrooms than the number of rooms.\n",
    "df7[df7.bath>df7.BHK+2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "0f3dce8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#removing all of these outliers\n",
    "df8=df7[df7.bath<df7.BHK+2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "125c86d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7251, 7)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df8.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "2b561c19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  BHK\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4\n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3\n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3\n",
       "3  1st Block Jayanagar      1200.0   2.0  130.0    3\n",
       "4  1st Block Jayanagar      1235.0   2.0  148.0    2"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#droppin two colums price_per_sqft and size\n",
    "df9=df8.drop([\"price_per_sqft\",\"size\"], axis=1)\n",
    "df9.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "fbd4dcb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7251, 5)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "32a2a427",
   "metadata": {},
   "source": [
    "After all these clean ups we are now left with 7251 records and 5 fields."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25cb3fcf",
   "metadata": {},
   "source": [
    "As we know that we cann't model any dataset with a field with text values, we  need to take the dummis of location feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "0f844597",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>7th Phase JP Nagar</th>\n",
       "      <th>8th Phase JP Nagar</th>\n",
       "      <th>9th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "      <th>other</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 242 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   1st Block Jayanagar  1st Phase JP Nagar  2nd Phase Judicial Layout  \\\n",
       "0                 True               False                      False   \n",
       "1                 True               False                      False   \n",
       "2                 True               False                      False   \n",
       "3                 True               False                      False   \n",
       "4                 True               False                      False   \n",
       "\n",
       "   2nd Stage Nagarbhavi  5th Block Hbr Layout  5th Phase JP Nagar  \\\n",
       "0                 False                 False               False   \n",
       "1                 False                 False               False   \n",
       "2                 False                 False               False   \n",
       "3                 False                 False               False   \n",
       "4                 False                 False               False   \n",
       "\n",
       "   6th Phase JP Nagar  7th Phase JP Nagar  8th Phase JP Nagar  \\\n",
       "0               False               False               False   \n",
       "1               False               False               False   \n",
       "2               False               False               False   \n",
       "3               False               False               False   \n",
       "4               False               False               False   \n",
       "\n",
       "   9th Phase JP Nagar  ...  Vishveshwarya Layout  Vishwapriya Layout  \\\n",
       "0               False  ...                 False               False   \n",
       "1               False  ...                 False               False   \n",
       "2               False  ...                 False               False   \n",
       "3               False  ...                 False               False   \n",
       "4               False  ...                 False               False   \n",
       "\n",
       "   Vittasandra  Whitefield  Yelachenahalli  Yelahanka  Yelahanka New Town  \\\n",
       "0        False       False           False      False               False   \n",
       "1        False       False           False      False               False   \n",
       "2        False       False           False      False               False   \n",
       "3        False       False           False      False               False   \n",
       "4        False       False           False      False               False   \n",
       "\n",
       "   Yelenahalli  Yeshwanthpur  other  \n",
       "0        False         False  False  \n",
       "1        False         False  False  \n",
       "2        False         False  False  \n",
       "3        False         False  False  \n",
       "4        False         False  False  \n",
       "\n",
       "[5 rows x 242 columns]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummy_loc= pd.get_dummies(df9[\"location\"])\n",
    "dummy_loc.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "665d1f9b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>7th Phase JP Nagar</th>\n",
       "      <th>8th Phase JP Nagar</th>\n",
       "      <th>9th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "      <th>other</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 242 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   1st Block Jayanagar  1st Phase JP Nagar  2nd Phase Judicial Layout  \\\n",
       "0                    1                   0                          0   \n",
       "1                    1                   0                          0   \n",
       "2                    1                   0                          0   \n",
       "3                    1                   0                          0   \n",
       "4                    1                   0                          0   \n",
       "\n",
       "   2nd Stage Nagarbhavi  5th Block Hbr Layout  5th Phase JP Nagar  \\\n",
       "0                     0                     0                   0   \n",
       "1                     0                     0                   0   \n",
       "2                     0                     0                   0   \n",
       "3                     0                     0                   0   \n",
       "4                     0                     0                   0   \n",
       "\n",
       "   6th Phase JP Nagar  7th Phase JP Nagar  8th Phase JP Nagar  \\\n",
       "0                   0                   0                   0   \n",
       "1                   0                   0                   0   \n",
       "2                   0                   0                   0   \n",
       "3                   0                   0                   0   \n",
       "4                   0                   0                   0   \n",
       "\n",
       "   9th Phase JP Nagar  ...  Vishveshwarya Layout  Vishwapriya Layout  \\\n",
       "0                   0  ...                     0                   0   \n",
       "1                   0  ...                     0                   0   \n",
       "2                   0  ...                     0                   0   \n",
       "3                   0  ...                     0                   0   \n",
       "4                   0  ...                     0                   0   \n",
       "\n",
       "   Vittasandra  Whitefield  Yelachenahalli  Yelahanka  Yelahanka New Town  \\\n",
       "0            0           0               0          0                   0   \n",
       "1            0           0               0          0                   0   \n",
       "2            0           0               0          0                   0   \n",
       "3            0           0               0          0                   0   \n",
       "4            0           0               0          0                   0   \n",
       "\n",
       "   Yelenahalli  Yeshwanthpur  other  \n",
       "0            0             0      0  \n",
       "1            0             0      0  \n",
       "2            0             0      0  \n",
       "3            0             0      0  \n",
       "4            0             0      0  \n",
       "\n",
       "[5 rows x 242 columns]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummy_loc=dummy_loc.astype(\"int\")\n",
    "dummy_loc.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "ea48c363",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>7th Phase JP Nagar</th>\n",
       "      <th>8th Phase JP Nagar</th>\n",
       "      <th>9th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 241 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   1st Block Jayanagar  1st Phase JP Nagar  2nd Phase Judicial Layout  \\\n",
       "0                    1                   0                          0   \n",
       "1                    1                   0                          0   \n",
       "2                    1                   0                          0   \n",
       "3                    1                   0                          0   \n",
       "4                    1                   0                          0   \n",
       "\n",
       "   2nd Stage Nagarbhavi  5th Block Hbr Layout  5th Phase JP Nagar  \\\n",
       "0                     0                     0                   0   \n",
       "1                     0                     0                   0   \n",
       "2                     0                     0                   0   \n",
       "3                     0                     0                   0   \n",
       "4                     0                     0                   0   \n",
       "\n",
       "   6th Phase JP Nagar  7th Phase JP Nagar  8th Phase JP Nagar  \\\n",
       "0                   0                   0                   0   \n",
       "1                   0                   0                   0   \n",
       "2                   0                   0                   0   \n",
       "3                   0                   0                   0   \n",
       "4                   0                   0                   0   \n",
       "\n",
       "   9th Phase JP Nagar  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                   0  ...            0                     0   \n",
       "1                   0  ...            0                     0   \n",
       "2                   0  ...            0                     0   \n",
       "3                   0  ...            0                     0   \n",
       "4                   0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "2                   0            0           0               0          0   \n",
       "3                   0            0           0               0          0   \n",
       "4                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "2                   0            0             0  \n",
       "3                   0            0             0  \n",
       "4                   0            0             0  \n",
       "\n",
       "[5 rows x 241 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#to aviod multicollinearity trap let's drop one column\n",
    "dummy_loc=dummy_loc.drop([\"other\"],axis=1)\n",
    "dummy_loc.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "f7d7e0f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#concatenation of df9 and dummy_loc.\n",
    "df10=pd.concat([df9,dummy_loc],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "3fd4e525",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 246 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  BHK  1st Block Jayanagar  \\\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4                    1   \n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3                    1   \n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3                    1   \n",
       "3  1st Block Jayanagar      1200.0   2.0  130.0    3                    1   \n",
       "4  1st Block Jayanagar      1235.0   2.0  148.0    2                    1   \n",
       "\n",
       "   1st Phase JP Nagar  2nd Phase Judicial Layout  2nd Stage Nagarbhavi  \\\n",
       "0                   0                          0                     0   \n",
       "1                   0                          0                     0   \n",
       "2                   0                          0                     0   \n",
       "3                   0                          0                     0   \n",
       "4                   0                          0                     0   \n",
       "\n",
       "   5th Block Hbr Layout  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                     0  ...            0                     0   \n",
       "1                     0  ...            0                     0   \n",
       "2                     0  ...            0                     0   \n",
       "3                     0  ...            0                     0   \n",
       "4                     0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "2                   0            0           0               0          0   \n",
       "3                   0            0           0               0          0   \n",
       "4                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "2                   0            0             0  \n",
       "3                   0            0             0  \n",
       "4                   0            0             0  \n",
       "\n",
       "[5 rows x 246 columns]"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df10.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "82af9c1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>BHK</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 245 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  price  BHK  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0  428.0    4                    1                   0   \n",
       "1      1630.0   3.0  194.0    3                    1                   0   \n",
       "2      1875.0   2.0  235.0    3                    1                   0   \n",
       "3      1200.0   2.0  130.0    3                    1                   0   \n",
       "4      1235.0   2.0  148.0    2                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "2                          0                     0                     0   \n",
       "3                          0                     0                     0   \n",
       "4                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                   0  ...            0                     0   \n",
       "1                   0  ...            0                     0   \n",
       "2                   0  ...            0                     0   \n",
       "3                   0  ...            0                     0   \n",
       "4                   0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "2                   0            0           0               0          0   \n",
       "3                   0            0           0               0          0   \n",
       "4                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "2                   0            0             0  \n",
       "3                   0            0             0  \n",
       "4                   0            0             0  \n",
       "\n",
       "[5 rows x 245 columns]"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#As we already have all the location data points in dummies, so can now drop location feature from df10.\n",
    "df11=df10.drop([\"location\"],axis=1)\n",
    "df11.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "8398aa11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>BHK</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 244 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  BHK  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0    4                    1                   0   \n",
       "1      1630.0   3.0    3                    1                   0   \n",
       "2      1875.0   2.0    3                    1                   0   \n",
       "3      1200.0   2.0    3                    1                   0   \n",
       "4      1235.0   2.0    2                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "2                          0                     0                     0   \n",
       "3                          0                     0                     0   \n",
       "4                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  6th Phase JP Nagar  ...  Vijayanagar  \\\n",
       "0                   0                   0  ...            0   \n",
       "1                   0                   0  ...            0   \n",
       "2                   0                   0  ...            0   \n",
       "3                   0                   0  ...            0   \n",
       "4                   0                   0  ...            0   \n",
       "\n",
       "   Vishveshwarya Layout  Vishwapriya Layout  Vittasandra  Whitefield  \\\n",
       "0                     0                   0            0           0   \n",
       "1                     0                   0            0           0   \n",
       "2                     0                   0            0           0   \n",
       "3                     0                   0            0           0   \n",
       "4                     0                   0            0           0   \n",
       "\n",
       "   Yelachenahalli  Yelahanka  Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0               0          0                   0            0             0  \n",
       "1               0          0                   0            0             0  \n",
       "2               0          0                   0            0             0  \n",
       "3               0          0                   0            0             0  \n",
       "4               0          0                   0            0             0  \n",
       "\n",
       "[5 rows x 244 columns]"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#takeing two new variables X and Y as our predictor and target variable respectively so as to train and test our model.\n",
    "X= df11.drop([\"price\"], axis=1)\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "52036fff",
   "metadata": {},
   "source": [
    "Here we have all the independent variables."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "e00b3d36",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y=df11[\"price\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ccf122aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    428.0\n",
       "1    194.0\n",
       "2    235.0\n",
       "3    130.0\n",
       "4    148.0\n",
       "Name: price, dtype: float64"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "470ef255",
   "metadata": {},
   "source": [
    "Y only has now price column i.e our target variable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "60d61943",
   "metadata": {},
   "source": [
    "## Model building:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "98d3b233",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10af8d85",
   "metadata": {},
   "source": [
    "Here we have divided our dependent and independent variable into train and test parts, where the test size is 20% meaning 20% of the data will be used for testing and rest 80% will be used in training the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "d6281ef5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.845227769787429"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Now we will use linear regression for model building.\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lr=LinearRegression()\n",
    "lr.fit(X_train,Y_train)\n",
    "lr.score(X_test,Y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7280b356",
   "metadata": {},
   "source": [
    "So our model is giving a R-Squared value of ~0.845. That means our model gives a good fit to the unseen dataset and 84.5% variability in the predictor variable is explained by the features."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "681222b7",
   "metadata": {},
   "source": [
    "### Using Cross-Validation method to test the accuracy of our Linear Regression Model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "00fb3bbc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.82430186, 0.77166234, 0.85089567, 0.80837764, 0.83653286])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import ShuffleSplit\n",
    "from sklearn.model_selection import cross_val_score\n",
    "cv=ShuffleSplit(n_splits=5, test_size=0.2,random_state=0)\n",
    "cross_val_score(lr,X,Y, cv=cv)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "d7a931ee",
   "metadata": {},
   "source": [
    "From this cross_val_score it's clear that our model will perform well enough on unseen data as it's giving score over 80% in most of the iterations."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9125c018",
   "metadata": {},
   "source": [
    "## Property price prediction:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "1af66bc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_price(location,total_sqft,bath,BHK):\n",
    "    index_location=np.where(X.columns==location)[0][0]\n",
    "    x=np.zeros(len(X.columns))\n",
    "    x[0]=total_sqft\n",
    "    x[1]=bath\n",
    "    x[2]=BHK\n",
    "    if index_location>=0:\n",
    "        x[index_location]=1\n",
    "    return lr.predict([x])[0]\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "1593afda",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Aniket Das\\anaconda3\\New folder\\Lib\\site-packages\\sklearn\\base.py:464: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "53.35838809775723"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict_price(\"Whitefield\",1000,2,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "b8bdc5f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Aniket Das\\anaconda3\\New folder\\Lib\\site-packages\\sklearn\\base.py:464: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "48.23042495140032"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict_price(\"Bellandur\",1000,2,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f459291e",
   "metadata": {},
   "source": [
    "## Conclusion:\n",
    "\n",
    "In conclusion, the developed linear regression model stands as a valuable tool for predicting Bangalore real estate prices, leveraging data acquired from Kaggle. Through data cleaning, feature engineering, and outlier handling, the model demonstrates robustness in capturing key factors influencing property values. Moving forward, continuous refinement and validation of the model against new data will be crucial to ensure its reliability and effectiveness in assisting real estate stakeholders with pricing decisions in the dynamic Bangalore market."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1aae5004",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
