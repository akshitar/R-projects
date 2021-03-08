rm(list=ls())
v <- View
require(data.table)
require(randomForest)
require(ggplot2)
require(plyr)
require(dplyr)
set.seed(1234)

# Read data
data <- fread('/Users/z002k4h/Downloads/conversion.csv', stringsAsFactors = TRUE, header = TRUE)

head(data)
# ~country : user country based on the IP address
# ~age : user age. Self reported at the sign up time
# ~new_user: whether the user created the account suring session or already had an
# account
# ~source: marketing channel source:
# 1. Ads: came to the site clicking on an advertisement
# 2. SEO: came to the site by clicking on search results
# 3. Direct: came to the site by directly typing URL
# ~total_pages_visited: number of total pages visited during that session. 
# Proxy for engagement on site
# ~converted: this is our label. 1 means they converted, 0 means left without
# buying. Goal is to increae conversion

summary(data)
# Quick observations:
# ~ site is probably a US site, although it does have a large China base
# ~ user base is pretty young
# ~ conversion rate is around 3%

# Let's look at age a little more closely
sort(unique(data$age))
# Well 111 and 123 seem unrealistic. How many users do we have exactly ?
subset(data, age > 80)

# Only two users. In this case let's remove those rows
data <- subset(data, age < 80)

# Before we start building the model, let's start by getting a sense of the data
data_country <- data %>% group_by(country) %>% summarise(conversion_rate = mean(converted))
ggplot(data = data_country, aes(x = country, y = conversion_rate)) +
  geom_col(aes(fill = country))
# Clearly looks like Chinese convert at a lower rate that other countries

data_pages <- data %>% group_by(total_pages_visited) %>% summarise(conversion_rate = mean(converted))
qplot(x = total_pages_visited, y = conversion_rate, data = data_pages, geom = 'line')
# As a user visits more pages or rather spends more time on the site,
# more likely he is to convert

data_source <- data %>% group_by(source) %>% summarise(conversion_rate = mean(converted))
ggplot(data=data_source, aes(x = source, y = conversion_rate)) +
  geom_col(aes(fill = source))
# Slighlty more for ads although more proportion of SEO users in our data

data$new_user <- as.factor(data$new_user)
data_newuser <- data %>% group_by(new_user) %>% summarise(conversion_rate = mean(converted))
ggplot(data = data_newuser, aes(x = new_user, y = conversion_rate)) + 
  geom_col(aes(fill=new_user))
# New users tend not to convert as compared to ones who already had an account

data_age <- data %>% group_by(age) %>% summarise(conversion_rate = mean(converted))
qplot(x = age, y = conversion_rate, data = data_age, geom = 'line')
# Looks the user base which converts tends to be young

# Let's build a model to predict conversion rate. Outcome is binary 
# , we care about insights to give product and marketing team.
# We are going to pick randomForest to predict CR because it usually requires
# very little time to optimize and is strong with outliers, irrelevent 
# variables, continuous and discrete variables. We will use PDP plots and 
# variable importance plots to get insights
data$converted <- as.factor(data$converted)

# Let's split train and test
train_sample <- sample(nrow(data), size = 0.66*nrow(data))
train_data <- data[train_sample,] 
test_data <- data[!train_sample,]
rf <- randomForest(x = train_data[, !c('converted')], y = train_data$converted, 
                   xtest = test_data[, !c('converted')], ytest = test_data$converted, ntree = 100,
                   mtry = 3, keep.forest = TRUE)
# The train and test error are pretty similar so we are confident we are not
# overfitting. Error is pretty low. We started with a 97% accuracy (if we had classified 
# everything as non-converted). So ~98.6% is good but nothing shocking. Indeed,
# 30% of conversion were predicted as non-converted.
# If we cared about the very best possible accuracy or minimizing FP/FN, we could 
# find the best cut off point. But in this case it's not particularly relevant,
# we are okay with 0.5 cutoff used internally by RF to make prediction

# After building the model and checking that the model predicts well, let's now 
# extract insights out of it
varImpPlot(rf)
# It looks like the total pages visited is the most important by far. But unfortunately 
# for us it's the least actionable. People visit many pages because they already want to 
# buy. Let's remove that variable andbuild the model again. Since we don't have this important
# variable anymore, let's change the weights a little bit to make sure we classsify something
# as 1
rf = randomForest(x = train_data[, !c('total_pages_visited', 'converted')], y = train_data$converted, 
                  xtest = test_data[, !c('total_pages_visited', 'converted')], ytest = test_data$converted,
                  ntree = 100, mtry = 3, keep.forest = TRUE, classwt = c(0.7, 0.3))
# Accuracy went down, but that's fine.
varImpPlot(rf)
# New user is now the most important, source on the other hand doesn't seem to matter

# Let's check the PDP plots for the 4 variable
op <- par(mfrow=c(2,2))
partialPlot(x=rf, pred.data = train_data, x.var = new_user, 1)
partialPlot(x=rf, pred.data = train_data, x.var = country, 1)
partialPlot(x=rf, pred.data = train_data, x.var = age, 1)
partialPlot(x=rf, pred.data = train_data, x.var = source, 1)
par(op)

# This shows that
# ~ Users with an old account are much better than new ones
# ~ China is really bad at converting overall
# ~ Site works well for young people and gets worse >30 yrs
# ~ Source is irrelevant

# Let's build a simple decision tree and check the 2/3 important segments
tree = rpart(formula = data$converted ~ ., data = data[, !c('total_pages_visited', 'converted')],
              control = rpart.control(maxdepth = 4), parms = list(prior=c(0.7, 0.3)))
rpart.plot(tree)

# Final conclusions:
# ~ Site is working well for young users. We can advertise more to a younger user base
# ~ Althought we only have few users from Germany, they do seem the site be working well
# for them in terms for conversion. We should be marketing more to Germans
# ~ Users with existing accounts seem to be converting more. We could send them emails with
# offers to get them back on the site
# ~ Maybe something is wrong with the Chinese version of the site ? Maybe poor translation,
# , maybe some payment issue, security issue. We could take a look and fix it
# ~ Why aren't more older people buying ?