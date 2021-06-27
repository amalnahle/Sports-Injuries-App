import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.arima.model import ARIMA
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statistics
import re
import base64
import json
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import openpyxl


#read all sheets
#df = pd.read_excel(r'C:\Users\USER\OneDrive - American University of Beirut\Desktop\Healthcare Project\Sports Injuries\Sports Injuries1.xlsx', sheet_name=None, header=0, squeeze= True)
url2 = 'https://drive.google.com/file/d/1vJlVWXMtlKAq9PafnRDrW_EIgMuRmmXh/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
df = pd.read_excel(path2, sheet_name=None, header=0, squeeze= True, engine='openpyxl')

#@st.cache
#def load_data(path):
   #df= pd.read_excel(path, sheet_name=None, header=0, squeeze= True, engine='openpyxl')
   #return df
#df= load_data(r'C:\Users\USER\OneDrive - American University of Beirut\Desktop\Healthcare Project\Sports Injuries\Sports Injuries1.xlsx')


#name the sheets
by_sport = df['Sport']
by_diagnosis = df['Diagnosis']
by_gender = df['Gender']
by_age = df['Age']
by_Ethnicity = df['Ethnicity']
by_location = df['Location']
cost = df['Cost']
gender_perc = df['Gender %']
by_sport2= df['Sport2']


#pivoting and converting dates - Injuries per sport
By_Sport = pd.melt(by_sport, id_vars=['Sport'], var_name='Year', value_name='Nb of Injuries').sort_values('Sport')
By_Sport['Year'] = By_Sport['Year'].apply(lambda x: str(x))
By_Sport= By_Sport.sort_values('Year')

By_Sport2 = pd.melt(by_sport2, id_vars=['Sport'], var_name='Year', value_name='Nb of Injuries').sort_values('Sport')
By_Sport2['Year']= pd.to_datetime(By_Sport2['Year'], format= '%Y')
By_Sport2= By_Sport2.sort_values('Year')
By_Sport2= By_Sport2.loc[By_Sport2['Year'] != '2020']
By_Sport2= By_Sport2.set_index('Year')

#print(By_Sport2)
#print(By_Sport2.dtypes)

#### Cost per Sport
Cost = pd.melt(cost, id_vars=['Sport'], var_name='Year', value_name='Cost').sort_values('Sport')
Cost['Year'] = Cost['Year'].apply(lambda x: str(x))
Cost= Cost.sort_values('Year')

### Gender
#across the years 
By_Gender = pd.melt(by_gender, id_vars=['Gender'], var_name='Year', value_name='Nb of Injuries').sort_values('Gender')
By_Gender['Year'] = By_Gender['Year'].apply(lambda x: str(x))
By_Gender= By_Gender.sort_values('Year')

gender_perc = pd.melt(gender_perc, id_vars=['Gender'], var_name='Year', value_name='Percent of Injuries').sort_values('Gender')
gender_perc['Year'] = gender_perc['Year'].apply(lambda x: str(x))
gender_perc= gender_perc.sort_values('Year')

#### Diagnosis
By_Diagnosis = pd.melt(by_diagnosis, id_vars=['Diagnosis'], var_name='Year', value_name='Nb of Injuries').sort_values('Diagnosis')
By_Diagnosis['Year'] = By_Diagnosis['Year'].apply(lambda x: str(x))
By_Diagnosis= By_Diagnosis.sort_values('Year')

#### Age
By_Age = pd.melt(by_age, id_vars=['Age'], var_name='Year', value_name='Nb of Injuries').sort_values('Age')
By_Age['Year'] = By_Age['Year'].apply(lambda x: str(x))

#### Ethnicity
By_Ethnicity = pd.melt(by_Ethnicity, id_vars=['Ethnicity'], var_name='Year', value_name='Nb of Injuries').sort_values('Ethnicity')
By_Ethnicity['Year'] = By_Ethnicity['Year'].apply(lambda x: str(x))
By_Ethnicity= By_Ethnicity.sort_values('Year')

#### MAPPING
#Location Table
By_Location = pd.melt(by_location, id_vars=['Location'], var_name='Year', value_name='Nb of Injuries').sort_values('Location')
By_Location['Year'] = By_Location['Year'].apply(lambda x: str(x))
By_Location= By_Location.sort_values('Year')

new_zealand = json.load(open('nz_region.geojson', 'r'))

state_id_map= {}
for feature in new_zealand['features']:
    feature['id']= feature['properties']['REGC2016']
    state_id_map[feature['properties']['REGC2016_N']]= feature['id']
   
By_Location['id']= By_Location['Location'].apply(lambda x: state_id_map[x])
By_Location['Scale']= np.log10(By_Location['Nb of Injuries'])

##covid-19 - 2020
df_2019= By_Sport.loc[By_Sport['Year'] == '2019']
df_2020= By_Sport.loc[By_Sport['Year'] == '2020']

###FORECAST MODEL###

#sport_arima = pd.read_excel(r'C:\Users\USER\OneDrive - American University of Beirut\Desktop\Healthcare Project\Sports Injuries\Sports Injuries1.xlsx', sheet_name=1, header=0, squeeze= True)

#sport_arima['Year']= pd.to_datetime(sport_arima['Year'], format= '%Y')
#sport_arima= sport_arima.set_index('Year')

#plt.plot(sport_arima)

#sport_arima_diff1 = sport_arima.diff().fillna(sport_arima)
#plt.plot(sport_arima_diff1)

#sport_arima_diff2 = sport_arima_diff1.diff().fillna(sport_arima_diff1)
#plt.plot(sport_arima_diff2)

#plot_acf(sport_arima_diff1)
#plt.show()

#plot_pacf(sport_arima_diff1, lags=5)
#plt.show()

# fit model
##model = ARIMA(sport_arima, order=(1,1,1)).fit()

#summary of fit model
##print(model.summary())

# line plot of residuals
#residuals = sport_arima(model_fit.resid)
#residuals.plot()
#pyplot.show()
# density plot of residuals
#residuals.plot(kind='kde')
#pyplot.show()
# summary stats of residuals
#print(residuals.describe())

# split into train and test sets
#X = sport_arima.values
#size = int(len(X) * 0.66)
#train, test = X[0:size], X[size:len(X)]
#history = [x for x in train]
#predictions = list()
## walk-forward validation
#for t in range(len(test)):
#	model = ARIMA(history, order=(1,1,1))
#	model_fit = model.fit()
#	output = model_fit.forecast()
#	yhat = output[0]
#	predictions.append(yhat)
#	obs = test[t]
#	history.append(obs)
#	print('predicted=%f, expected=%f' % (yhat, obs))
## evaluate forecasts
#rmse = np.sqrt(mean_squared_error(test, predictions))
#print('Test RMSE: %.3f' % rmse)

############ STREAMLLIT APP ################
st.set_page_config(page_title="Sports Injuries (New Zealand)", page_icon=':football:', layout='wide')

#home page image

image = 'https://www.bassairpinia.it/wp-content/uploads/2019/02/sport1.png'
image1= 'https://admin.kowsarpub.com/cdn/serve/314b7/58b3f2a4b33a771a3e8de06b714987bd533c51aa/shutterstock_639227380(1).png'
image2= 'https://www.nicepng.com/png/full/102-1020171_arrow-vector-blue-arrow-abstract.png'

#side bar menu
selectbox = st.sidebar.selectbox('MENU', ('Home Page', 'Dashboard'))

#Home Page
if selectbox == 'Home Page':
    
    st.markdown(f"<div style='text-align: center; font-size: 40px; font-weight: bold; background-color:#14AEC0; font-family:Montserrat;'>  Accident Compensation Corporation (ACC) <div style='text-align: center; font-size: 30px; color: #DFE6E6; font-weight: light; background-color:#14AEC0; font-family:Times New Roman;'> | Sports Injuries in New Zealand | </div>", unsafe_allow_html=True)
    
    g1, g5, g2, g3= st.beta_columns([1.3, 0.7, 3, 1.6])
    
    with g2:
        
        #upload image
        st.image(image, width= 540, output_format='JPEG', channels="BGR")

    with g3:
        
        #upload files
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        upload_file= st.file_uploader('UPLOAD DATA', type='xlsx')
        

    st.markdown(f"<div style='text-align: center; font-size: 30px; color:#14AEC0; font-weight: bold; background-color:#14AEC0; font-family:Montserrat;'>  A  </div>", unsafe_allow_html=True)
    
    with g1:
         
        #upload image
        st.markdown("")
        st.image(image2, width= 345, output_format='JPEG', channels="BGR")
    

        
#Dashboard Page

if selectbox == 'Dashboard':
    
    #dashboard title
    st.markdown(f"<p style='text-align: center; font-size: 20px; font-weight: bold; background-color:#14AEC0; font-family:Montserrat;'>  Sports Injuries Dashboard For New Zealand</p>", unsafe_allow_html=True)
    
    g1, g2, g3= st.beta_columns([1.5, 2.2, 3])
    
    with g2:
        st.markdown(":table_tennis_paddle_and_ball:")
    
    with g1:
        
        st.markdown(f"<p style='text-align: left ;font-size: 25px; font-family:Montserrat;'> OVERVIEW BY SPORT </p>", unsafe_allow_html=True)
    
    
    g1, g2, g3= st.beta_columns([1, 1, 2.5])
        
    with g1:
        
        sports_options= By_Sport['Sport'].unique().tolist()
        sports= st.selectbox('Select Sport', sports_options)
        filter_by_sport= By_Sport[By_Sport['Sport']== sports]
        Cost_filtered= Cost[Cost['Sport']== sports]
        filtered_2019= df_2019[df_2019['Sport']== sports]
        filtered_2020= df_2020[df_2020['Sport']== sports]
        filter_by_sport2= By_Sport2[By_Sport2['Sport']== sports]
        
    with g2:
        
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        
    with g3:
    
        #### LINE CHART #####
        fig= px.line(filter_by_sport, x= filter_by_sport['Year'], y= filter_by_sport['Nb of Injuries'], color= filter_by_sport['Sport'], color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor=' #D9DAD9',
        plot_bgcolor=' #D9DAD9', legend_title_text="", showlegend= False, autosize=False,
        width=655,
        height=270,
        margin=dict(
            l=40,
            r=40,
            b=40,
            t=10,
            pad=4))
        fig.update_xaxes(title="", showgrid= False)
        fig.update_yaxes(title="Total nb of injuries", showgrid= False)
        fig.update_xaxes(title="Number of injuries throughout the years",
            title_font = {"size": 20, "family": 'Montserrat'})
        st.write(fig)
    
    with g2:
        
        #percent change
        perc_change= 100* (float(filtered_2020['Nb of Injuries']) - filtered_2019['Nb of Injuries']) / filtered_2019['Nb of Injuries']
        perc_change= "{:,.1f}%".format(statistics.mean(perc_change.values))
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'>  % change from 2019 to 2020 <div style='text-align: center; color: #14AEC0; font-size: 35px; font-weight: bold; font-family:Montserrat;'> {perc_change} </div>", unsafe_allow_html=True)
        
    with g2:
        
        ###FORECASTS (excluding year 2020)
        df2= filter_by_sport2.drop(['Sport'], axis= 1)
        model = ARIMA(df2, order=(1,1,1)).fit()
        arima_predict= model.predict('2021-01-01', type= 'levels')
        pred_value= "{:,.1f}".format(arima_predict[0])
        #print('Forecast', arima_predict)
        st.markdown('')
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat;background-color: #D9DAD9;'> Forecasted injuries in 2021 <div style='text-align: center; color: #14AEC0; font-size: 35px; font-weight: bold;font-family:Montserrat;'> {pred_value} </div>", unsafe_allow_html=True)
       
        
    with g1:
       
        # avg injuries
        Avg_injuries= statistics.mean(filter_by_sport['Nb of Injuries'])
        Avg_injuries= "{:,.1f}".format(Avg_injuries)
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Avg. number of injuries per year <div style='text-align: center; color: #14AEC0; font-size: 35px; font-weight: bold;font-family:Montserrat;'> {Avg_injuries} </div>", unsafe_allow_html=True)
        
    with g1:
       
        st.markdown('')
        Avg_cost= statistics.mean(Cost_filtered['Cost'])
        Avg_cost1= "${:,.1f}".format(Avg_cost)
        nan= "NA"
        if Avg_cost1 =="$nan":
            st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Avg. cost per year <div style='text-align: center; color: #14AEC0 ;font-size: 35px; font-weight: bold;font-family:Montserrat;'> {nan} </div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Avg. cost per year <div style='text-align: center; color: #14AEC0 ;font-size: 35px; font-weight: bold;font-family:Montserrat;'> {Avg_cost1} </div>", unsafe_allow_html=True)
    
    #with g3:
        #fig7 = px.bar(By_Gender, x="Year", y="Nb of Injuries", color="Gender", barmode = 'stack', color_discrete_sequence=px.colors.qualitative.Pastel)
        #fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)', legend_title_text="")
        #fig7.update_xaxes(title="")
        #st.write(fig7)    
    
    #Dashboard section 2
    
    st.markdown("")
    g1, g2, g3= st.beta_columns([1.5, 2.4, 3])
    
    with g2:
        st.markdown(":calendar:")
    
    with g1:
        st.markdown(f"<p style='text-align: left ;font-size: 25px; font-family:Montserrat; '> OVERVIEW BY YEAR </p>", unsafe_allow_html=True)
    
    st.markdown("")
    
    g1, g2, g0, g3= st.beta_columns([1, 2, 0.9, 2])
    
    with g1:
    
        #filters
        year_options= By_Sport['Year'].unique().tolist()
        years= st.selectbox('Select Year', year_options)
        filter_by_year= By_Sport[By_Sport['Year']== years]
        Cost_filter_yr= Cost[Cost['Year']== years]
        diag= By_Diagnosis[By_Diagnosis['Year']== years]
        age= By_Age[By_Age['Year']== years]
        eth=By_Ethnicity[By_Ethnicity['Year']== years]
        Location=By_Location[By_Location['Year']== years]
        gender_filter= gender_perc[gender_perc['Year']== years]
        gender_male= gender_filter.loc[gender_filter['Gender'] == 'Male']
        gender_female= gender_filter.loc[gender_filter['Gender'] == 'Female']
        
    with g1:
        
        #gender
        male_perc= "{:,.1f}%".format(statistics.mean(gender_male['Percent of Injuries'].values))
        female_perc= "{:,.1f}%".format(statistics.mean(gender_female['Percent of Injuries'].values))
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> % Male injuries <div style='text-align: center; color: #14AEC0; font-size: 35px; font-weight: bold;font-family:Montserrat;'> {male_perc} </p>", unsafe_allow_html=True)
    
    with g1: 
    
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> % Female injuries <div style='text-align: center; color: #14AEC0; font-size: 35px; font-weight: bold;font-family:Montserrat;'> {female_perc} </p>", unsafe_allow_html=True)
            
    with g2:
        
        #sports barchart
        st.markdown("")
        fig1= px.bar(filter_by_year, y= filter_by_year['Sport'], x= filter_by_year['Nb of Injuries'], orientation='h', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')
        fig1.update_xaxes(title="Number of injuries across sports", showgrid= False,
            title_font = {"size": 20, "family": 'Montserrat'}, title_standoff = 25)
        fig1.update_yaxes(title="", showgrid= False)
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        fig1.update_layout(paper_bgcolor=' #D9DAD9',
        plot_bgcolor=' #D9DAD9')
        fig1.update_layout(autosize=False,
        width=590,
        height=255,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4))
        st.write(fig1)
    
    with g3:
        
        st.markdown("")
        #map
        fig4= px.choropleth(Location, locations= 'id', geojson= new_zealand, hover_data= Location.columns.tolist(), color= 'Nb of Injuries', hover_name='Location', color_continuous_scale= px.colors.sequential.Teal)
        fig4.update_geos(fitbounds='locations', visible=False)
        #fig4.update(layout_coloraxis_showscale=False)
        fig4.update_layout(geo=dict(bgcolor= '#D9DAD9'), paper_bgcolor=' #D9DAD9')
        fig4.update_layout(autosize=False, width=400, height=255, margin=dict(l=10,r=10,b=10,t=10,pad=4))
        fig4.update_layout(title="Number of injuries across regions", title_font = {"size": 20, "family": 'Montserrat'}, title_x=0.8, title_y=0.03)
        st.write(fig4)
        
    g1, g2, g3= st.beta_columns(3)
    
    with g1:
        
        #diagnosis pie chart
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 20px;font-family:Montserrat; background-color: #D9DAD9;'> % Injuries by diagnosis </div>", unsafe_allow_html=True)
        fig2= px.pie(diag, values= 'Nb of Injuries', names= 'Diagnosis', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(autosize=False,
        width=389,
        height=400,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4))
        fig2.update_layout(paper_bgcolor=' #D9DAD9')
        fig2.update_layout(legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.2))
        st.write(fig2)
        
    with g2:    
        
        #age pie chart with hole
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 20px;font-family:Montserrat; background-color: #D9DAD9;'> % Injuries by age </div>", unsafe_allow_html=True)
        fig5= go.Figure(data=[go.Pie(labels=age['Age'], values=age['Nb of Injuries'], hole=.3)])
        fig5.update_traces(marker_colors=px.colors.qualitative.Pastel)  
        fig5.update_layout(autosize=False, width=390, height=400, margin=dict(l=10, r=10, b=10, t=10,pad=4), paper_bgcolor=' #D9DAD9')
        fig5.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.2))
        st.write(fig5)
    
    with g3:
        
        #ethnicity pie chart with hole
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 20px; font-family:Montserrat; background-color: #D9DAD9;'> % Injuries by ethnicity </div>", unsafe_allow_html=True)
        fig6= go.Figure(data=[go.Pie(labels=eth['Ethnicity'], values=eth['Nb of Injuries'], hole=.3)])
        fig6.update_traces(marker_colors=px.colors.qualitative.Pastel)
        fig6.update_layout(autosize=False, width=389, height=400, margin=dict(l=10, r=10, b=10, t=10,pad=4), paper_bgcolor=' #D9DAD9')
        fig6.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.2))
        st.write(fig6)
        
    with g1:
        
        #diagnosis filter
        diag_options= By_Diagnosis['Diagnosis'].unique().tolist()
        diag_yr= st.selectbox('Select Diagnosis', diag_options)
        filter_by_diag= By_Diagnosis[(By_Diagnosis['Year']== years) & (By_Diagnosis['Diagnosis']== diag_yr)]    
        diag_injuries= statistics.mean(filter_by_diag['Nb of Injuries'])
        diag_injuries= "{:,.0f}".format(diag_injuries)
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Number of injuries in selected year <div style='text-align: center; color: #14AEC0 ;font-size: 35px; font-weight: bold;font-family:Montserrat;'> {diag_injuries} </div>", unsafe_allow_html=True)
        
    with g2:
        
        #age filter
        age_options= By_Age['Age'].unique().tolist()
        age_yr= st.selectbox('Select Age', age_options)
        filter_by_age= By_Age[(By_Age['Year']== years) & (By_Age['Age']== age_yr)]    
        age_injuries= statistics.mean(filter_by_age['Nb of Injuries'])
        age_injuries= "{:,.0f}".format(age_injuries)
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Number of injuries in selected year <div style='text-align: center; color: #14AEC0 ;font-size: 35px; font-weight: bold;font-family:Montserrat;'> {age_injuries} </div>", unsafe_allow_html=True)
        
    with g3:
        
        #ethnicity filter
        e_options= By_Ethnicity['Ethnicity'].unique().tolist()
        e_yr= st.selectbox('Select Ethnicity', e_options)
        filter_by_e= By_Ethnicity[(By_Ethnicity['Year']== years) & (By_Ethnicity['Ethnicity']== e_yr)]    
        e_injuries= statistics.mean(filter_by_e['Nb of Injuries'])
        e_injuries= "{:,.0f}".format(e_injuries)
        st.markdown(f"<div style='text-align: center; color: #000505; font-size: 15px;font-family:Montserrat; background-color: #D9DAD9;'> Number of injuries in selected year <div style='text-align: center; color: #14AEC0 ;font-size: 35px; font-weight: bold;font-family:Montserrat;'> {e_injuries} </div>", unsafe_allow_html=True)
        
        
