library("CCA")
library("dplyr")
library("ggplot2")
library("gstat")
library("sp")
library("spacetime")
library("STRbook")
library("tidyr")
library("grid")
library("gridExtra")

FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA_unique.csv'

data = read.csv(paste(FOLDER,NAME, sep=""), sep = ',', header = TRUE)
#data = read.csv(paste(FOLDER,NAME, sep=""), sep = ';', header = TRUE)[ ,c(3,5,6,7,9)]

#names(data) = c('time', 'lon', 'lat', 'dep', 'chl')
#data$year = substr(data$time, 7, 10)
#data$month = substr(data$time, 4, 5)
#data$day = substr(data$time, 1, 2)
#data$date = with(data, paste(year, month, day, sep = "-"))
data$date = as.Date(data$date)
data$t = as.numeric(format(data$date, "%j")) - 209

## -----------------------------------------------------------
spat_av <- group_by(data, lat, lon) %>%    # group by lon-lat
  summarise(mu_emp = mean(chl))     # mean for each lon-lat

lat_means <- ggplot(spat_av) +
  geom_point(aes(lat, mu_emp)) +
  xlab("Latitude (deg)") +
  ylab("Chlorophyll Concentration (mg)") + theme_bw()

lon_means <- ggplot(spat_av) +
  geom_point(aes(lon, mu_emp)) +
  xlab("Longitude (deg)") +
  ylab("Chlorophyll Concentration (mg)") + theme_bw()

print(lat_means)
print(lon_means)

## -----------------------------------------------------------
Tmax_av <- group_by(data, date) %>%
  summarise(meanTmax = mean(chl))

gTmaxav <-
  ggplot() +
  geom_line(data = data,aes(x = date, y = chl),
            colour = "blue", alpha = 0.4) +
  geom_line(data = Tmax_av, aes(x = date, y = meanTmax)) +
  xlab("Month") + ylab("Chlorophyll Concentration (mg)") +
  theme_bw()
print(gTmaxav)
## -----------------------------------------------------------
lm1 <- lm(chl ~ lon + lat + t + I(t^2), data = data) # fit a linear model
data$residuals <- residuals(lm1)             # store the residuals

## ------------------------------------------------------------------------
spat_df <- filter(data, t == 1) %>% # lon/lat coords of stations
  select(lon, lat)  %>%   # select lon/lat only
  arrange(lon, lat)       # sort ascending by lon/lat
m <- nrow(spat_av)                  # number of stations

## ------------------------------------------------------------------------
X <- select(data, lon, lat, residuals, t) %>% # select columns
  spread(t, residuals) %>%                 # make time-wide
  select(-lon, -lat) %>%                   # drop coord info
  t()                                      # make space-wide

## ------------------------------------------------------------------------
Lag0_cov <- cov(X, use = 'complete.obs')
Lag1_cov <- cov(X[-1, ], X[-nrow(X),], use = 'complete.obs')