# Analyzing New York City Taxi Data Using Apache Spark
## Introduction
With the rapid development of wireless communication technology and intelligent mobile terminal, collecting the track record of moving objects becomes simple and fast. Due to the fact that the New York taxi trajectory data has not only spatial attributes, but also contains a time attribute, it becomes the research subjects and advanced application for spatial temporal data mining. However, the hot zone of passengers pick-up points will change with time, and at the same time, the city’s available points will affect the passengers’ travel pattern. How to accurately find the concentration of passengers in different periods of time is the necessary condition to the recommend services for taxi drivers. 

Firstly, this report will analysis the New York Taxi trajectory data to get the results of how people commute in the city. As mentioned before, the taxi trajectory data is big data so that the report also utilize the Spark on ROGER to calculate the results. After that, the report will use the methods of spatial autocorrelation to come up with the hot zone recommendation for New York and provide suggestion for city administration and taxi drivers in terms of improvement of the taxi services in these areas.
## Data and Methodology
In this report, the New York taxi data containing over 14 million taxi records during January, 2013, is used to conduct the analysis. It is provided by the course of GEOG 479 in the Department of Geology. The taxi records of the data have detailed description of the taxi communication, including medallion, hack license, vendor id, rate code, store and fwd flag, pickup datetime, dropoff datetime, passenger count, trip time, trip distance, pickup longitude, pickup latitude, dropoff longitude, dropoff latitude, payment type, fare amount, surcharge, mta tax, tip amount, tolls amount, total amount. In this report, the descriptions of passenger count and pick-up location are mainly used to demonstrate the ridership in different traffic analysis zones with the help of the shapefile of the TAZ of New York. In future analysis, more research on the description of location and time periods can be conducted to analyze the change of taxi hot zones in different location and time periods. However, in this report, all the data will be considered to provide more general suggestion for taxi drivers and city administrators.
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Analyzing_New_York_City_Taxi_Data_Using_Apache_Spark/Capture.JPG" width="750"/>
  The raw data plotted on the map - messy and not informative
</p>
## Analysis of Results

## Conclusion

