##################################################################################
### Job Transitions Study: Why Do Some Employees Leave Firms When Changing Jobs?
### by Morgan Smart and Andrew Chamberlain
### Glassdoor Economic Research (glassdoor.com/research/)
### January 2017
##################################################################################

# Disable scientific notation
options(scipen=999)

# Load packages
library(MatchIt)
library(leaps)
library(MASS)
library(DAAG)
library(caret)
library(plyr)
library(randomForest)
library(stargazer)
library(gridExtra)
library(sandwich)
library(scales)



#############################
###### LEFT EMPLOYER ########
#############################

# import job history data
jobHistory <- read.csv("data.csv")

# Data cleaning and prep
jobHistory <- na.omit(jobHistory)
jobHistory <- jobHistory[-jobHistory$yearFounded<0,]
jobHistory <- jobHistory[-jobHistory$employeesTotalNum<1,]
jobHistory <- jobHistory[-jobHistory$yearFounded_new<0,]
jobHistory <- jobHistory[-jobHistory$employeesTotalNum_new<1,]

# Create employer size buckets
jobHistory$empSize_bucket <- ifelse(jobHistory$employeesTotalNum >= 1 & jobHistory$employeesTotalNum <= 9, 1,
                                    ifelse(jobHistory$employeesTotalNum >= 10 & jobHistory$employeesTotalNum <= 49, 2,
                                           ifelse(jobHistory$employeesTotalNum >= 50 & jobHistory$employeesTotalNum <= 249, 3,
                                                  ifelse(jobHistory$employeesTotalNum >= 250 & jobHistory$employeesTotalNum <= 999, 4,
                                                         ifelse(jobHistory$employeesTotalNum >= 1000 & jobHistory$employeesTotalNum <= 4999, 5,
                                                                ifelse(jobHistory$employeesTotalNum >= 5000 & jobHistory$employeesTotalNum <= 9999, 6,
                                                                       ifelse(jobHistory$employeesTotalNum >= 10000 & jobHistory$employeesTotalNum <= 99999, 7,
                                                                              ifelse(jobHistory$employeesTotalNum >= 100000, 8, NA
                                                                              ))))))))

jobHistory$empSize_bucket_new <- ifelse(jobHistory$employeesTotalNum_new >= 1 & jobHistory$employeesTotalNum_new <= 9, 1,
                                        ifelse(jobHistory$employeesTotalNum_new >= 10 & jobHistory$employeesTotalNum_new <= 49, 2,
                                               ifelse(jobHistory$employeesTotalNum_new >= 50 & jobHistory$employeesTotalNum_new <= 249, 3,
                                                      ifelse(jobHistory$employeesTotalNum_new >= 250 & jobHistory$employeesTotalNum_new <= 999, 4,
                                                             ifelse(jobHistory$employeesTotalNum_new >= 1000 & jobHistory$employeesTotalNum_new <= 4999, 5,
                                                                    ifelse(jobHistory$employeesTotalNum_new >= 5000 & jobHistory$employeesTotalNum_new <= 9999, 6,
                                                                           ifelse(jobHistory$employeesTotalNum_new >= 10000 & jobHistory$employeesTotalNum_new <= 99999, 7,
                                                                                  ifelse(jobHistory$employeesTotalNum_new >= 100000, 8, NA
                                                                                  ))))))))

jobHistory$startYear <- as.numeric(substr(jobHistory$startDate,1,4))
jobHistory$startYear_new <- as.numeric(substr(jobHistory$startDate_new,1,4))
jobHistory$logSalary <- log(jobHistory$avgSalary)
jobHistory$logSalary_new <- log(jobHistory$avgSalary_new)
 
# Get column names
jobHistory <- jobHistory[,-1] #remove de-duping column
names(jobHistory)
jobHistory <- na.omit(jobHistory)


# Create data frame for regressions: 5006 obs
df_all <- jobHistory[,c(3,6,8,10:25,28,30:50)]
df_all <- na.omit(df_all)
names(df_all)
df <- df_all[,-c(3,8,21,26)] # remove employeesTotalNum and avgSalary
names(df)
df$newMinusOldSalary <- df$logSalary_new - df$logSalary
df <- na.omit(df)
df <- df[,-c(18:31,33,35,37)]  # choose things about first job only

# Get if left emp or not
leftEmp <- df[which(df_all$leftEmployer == 1),] #3660 obs 
stayedEmp <- df[which(df_all$leftEmployer == 0),] #1346 obs

# ## Best predictors to use in regression ###
# features <- df[,-c(1,4:6)]
# # Correlated features?
# correlationMatrix <- cor(features)
# # summarize the correlation matrix
# print(correlationMatrix)
# # find attributes that are highly corrected (ideally >0.75)
# highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75, exact = TRUE, names=TRUE)
# print(highlyCorrelated) #ratingOverall and ratingSeniorManagement correlated (but not a problem because the regression will never include both) and ratingRecommendToFriend



#############################
##### SUMMARY TABLE ########
#############################

# Summary statistics table for regression analysis
stargazer(df_all, type = 'html', digits=2, out = '/Users/andrew.chamberlain/Google Drive/Research/Report 12 - Job Transitions Study/Data and Code/summary.html')

# Unique number of resumes.
length(unique(jobHistory$resumeId)) #4,592 resumes.

# Number of transitions with a pay raise vs. pay decline. 
length(which(jobHistory$logSalary_new - jobHistory$logSalary >= 0)) # Upward pay move = 3,146
length(which(jobHistory$logSalary_new - jobHistory$logSalary < 0)) # Upward pay move = 1,860

# Pay raise for job stayers.
mean(jobHistory$avgSalary_new[jobHistory$leftEmployer == 0] - jobHistory$avgSalary[jobHistory$leftEmployer == 0]) # Pay raise for job stayers. 
mean(jobHistory$avgSalary_new[jobHistory$leftEmployer == 0] - jobHistory$avgSalary[jobHistory$leftEmployer == 0]) / mean(jobHistory$avgSalary[jobHistory$leftEmployer == 0]) # Percent pay raise.

# Pay raise for job leavers. 
mean(jobHistory$avgSalary_new[jobHistory$leftEmployer == 1] - jobHistory$avgSalary[jobHistory$leftEmployer == 1]) # Pay raise for leavers. 
mean(jobHistory$avgSalary_new[jobHistory$leftEmployer == 1] - jobHistory$avgSalary[jobHistory$leftEmployer == 1]) / mean(jobHistory$avgSalary[jobHistory$leftEmployer == 1])

# Summary stats for job leavers only. 
stargazer(jobHistory[jobHistory$leftEmployer == 1, ], type = 'html', digits=2, out = '/Users/andrew.chamberlain/Google Drive/Research/Report 12 - Job Transitions Study/Data and Code/summary_leavers.html')


################################
##### DATA VISUALIZATIONS ######
################################

#7cb228 - GREEN
#2c84cc - BLUE


# # PDF of new minus old salary.
# ggplot(df_all, aes(x = newMinusOldSalary))+
#   ggtitle('Distribution of Pay Changes')+
#   labs(x='New Minus Old Salary', y='Frequency')+
#   theme(title = element_text(color='#2c84cc', size=14),
#         axis.text = element_text(size = 12),
#         plot.title = element_text(hjust = 0.5))+
#   scale_x_continuous(breaks = pretty_breaks(n=8),
#                      labels = scales::dollar)+
#   scale_y_continuous(labels = comma)+
#   geom_histogram(binwidth=15000, colour = '#7cb228', fill='#7cb228', alpha = 0.8)



# 2. PDF of length of first job.
ggplot(df_all, aes(x = jobLength_old))+
  ggtitle('Distribution of Months Spent at Original Job')+
  labs(x='Months', y='Frequency')+
  theme(title = element_text(color='#2c84cc', size=14),
        axis.text = element_text(size = 12),
        plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = pretty_breaks(n=10),
                     labels = comma)+
  scale_y_continuous(labels = comma)+
  geom_histogram(binwidth=3, colour = 'gray50', fill='#7cb228', alpha = 1.0)


# 4-1 PDF of salary for first job.
ggplot(df_all, aes(x = avgSalary))+
  ggtitle('Distribution of Annual Pay For Original Job')+
  labs(x='Average Annual Salary', y='Frequency')+
  theme(title = element_text(color='#2c84cc', size=14),
        axis.text = element_text(size = 12),
        plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = pretty_breaks(n=8),
                     labels = scales::dollar)+
  scale_y_continuous(labels = comma)+
  geom_histogram(binwidth=15000, colour = 'gray50', fill='#7cb228', alpha = 1.0)


# 4-2 PDF of salary for second job.
ggplot(df_all, aes(x = avgSalary_new))+
  ggtitle('Distribution of Annual Pay For New Job')+
  labs(x='Average Annual Salary', y='Frequency')+
  theme(title = element_text(color='#2c84cc', size=14),
        axis.text = element_text(size = 12),
        plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = pretty_breaks(n=8),
                     labels = scales::dollar)+
  scale_y_continuous(labels = comma)+
  geom_histogram(binwidth=15000, colour = 'gray50', fill='#7cb228', alpha = 1.0)



# Mean job length by industry sector.
by_sector <- aggregate(jobLength_old ~ gdSectorName, data = df_all, FUN = function(x) c(mn = mean(x), n = length(x)))
print(by_sector)
write.csv(by_sector, '/Users/andrew.chamberlain/Google Drive/Research/Report 12 - Job Transitions Study/Data and Code/job_length_by_sector.csv')


# Probability of leaving employer by industry sector.
by_sector_left <- aggregate(leftEmployer ~ gdSectorName, data = df_all, FUN = function(x) c(mn = mean(x), n = length(x)))
print(by_sector_left)
write.csv(by_sector_left, '/Users/andrew.chamberlain/Google Drive/Research/Report 12 - Job Transitions Study/Data and Code/by_sector_left.csv')



# Percent that left employer when they changed job. 
percentLeft <- mean(df_all$leftEmployer)




# Overlayed PDFs of stay vs. leave against income. 





# Overlayed PDFs of stay vs. leave against overall rating. 








# # Histograms of employee stats
# ggplot(df_all, aes(x = startYear))+ggtitle('First Job Start Year')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(binwidth=1,colour = '#2c84cc', fill='#2c84cc', alpha = .5)
# 
# ggplot(df_all, aes(x = avgSalary))+ggtitle('Average Salary Distribution of First Job')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(binwidth=15000,colour = '#2c84cc', fill='#2c84cc', alpha = .5)
# 
# ggplot(df_all, aes(x = jobLength_old))+ggtitle('Length of First Job (in months)')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(binwidth=1,colour = '#2c84cc', fill='#2c84cc', alpha = .5)
# 
# ggplot(df_all, aes(x = yearFounded))+ggtitle('First Job Employer\'s Year Founded')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(colour = '#2c84cc', fill='#2c84cc', alpha = .5)
# 
# 
# # Histograms of employer stats
# ggplot(df_all, aes(x = employeesTotalNum))+ggtitle('Employer Size Distribution')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(colour = '#7cb228', fill='#7cb228', alpha = .5, binwidth=100000)
# 
# ggplot(df_all, aes(x = empSize_bucket))+ggtitle('Employer Size (bucketed) Distribution')+
#   labs(x='', y='')+theme(title=element_text(color='#7cb228', size=10))+
#   geom_histogram(binwidth=1,colour = '#2c84cc', fill='#2c84cc', alpha = .5)



#############################
###### REGRESSION MODELS ####
#############################

# basic model: overall rating rating
fit1 <- lm(leftEmployer ~ ratingOverall,data = df)
summary(fit1)

# basic model: subratings
fit2 <- lm(leftEmployer ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance
           + ratingSeniorManagement + ratingCompensationAndBenefits, data = df)
summary(fit2)

# complex model: job properties, overall rating
fit3 <- lm(leftEmployer ~ ratingOverall + yearFounded + empSize_bucket
           + jobLength_old + startYear + logSalary + newMinusOldSalary, data = df)
summary(fit3)

# complex model: job properties, subratings
fit4 <- lm(leftEmployer ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance + ratingCompensationAndBenefits
           + ratingSeniorManagement + yearFounded + empSize_bucket + jobLength_old
           + startYear + logSalary + newMinusOldSalary, data = df)
summary(fit4)


# all model: overall rating
fit5 <- lm(leftEmployer ~ ratingOverall + yearFounded + empSize_bucket
           + jobLength_old + startYear + metro + jobTitle + gdSectorName
           + logSalary + newMinusOldSalary, data = df)
summary(fit5)

# all model: subratings
fit6 <- lm(leftEmployer ~ ratingCeo
           + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance + ratingCompensationAndBenefits
           + ratingSeniorManagement + yearFounded + empSize_bucket +
             jobLength_old + startYear + metro + jobTitle + gdSectorName
           + logSalary + newMinusOldSalary, data = df)
summary(fit6)



#############################
##### REGRESSION OUTPUT #####
#############################

# Calculate White standard errors ("heteroskedasticity robust") for LPM model.
vc_1 <- vcovHC(fit1, "HC1")
vc_2 <- vcovHC(fit2, "HC1")
vc_3 <- vcovHC(fit3, "HC1")
vc_4 <- vcovHC(fit4, "HC1")
vc_5 <- vcovHC(fit5, "HC1")
vc_6 <- vcovHC(fit6, "HC1")

se_1 <- sqrt(diag(vc_1))
se_2 <- sqrt(diag(vc_2))
se_3 <- sqrt(diag(vc_3))
se_4 <- sqrt(diag(vc_4))
se_5 <- sqrt(diag(vc_5))
se_6 <- sqrt(diag(vc_6))

# all data
stargazer(fit1, fit2, fit3, fit4, fit5, fit6,
          se = list(se_1, se_2, se_3, se_4, se_5, se_6),
          title = "Linear Probability Model Results",
          type = 'html',
          digits = 2,
          omit = c("jobTitle", "gdSectorName", "metro"),
          out = "/Users/andrew.chamberlain/Google Drive/Research/Report 12 - Job Transitions Study/Data and Code/results_robust.html")



#############################
##### EXPLORATION ###########
#############################

names(jobHistory)
data <- jobHistory
data$ratingOverall_dif <- data$ratingOverall_new - data$ratingOverall
data$avgSalary_dif <- data$avgSalary_new - data$avgSalary
data$sameSector <- ifelse(as.character(data$gdSectorName)==as.character(data$gdSectorName_new),1,0)
data$sameTitle <- ifelse(as.character(data$jobTitle)==as.character(data$jobTitle_new),1,0)
data$sameMetro <- ifelse(as.character(data$metro)==as.character(data$metro_new),1,0)

# everyone
par(mar = c(15,5,2,2))
boxplot(jobLength_old ~ gdSectorName, data = data, las=2, ylab = "First Job Length (in months)")
title("Job Length (in months) by Industry")

# left
boxplot(jobLength_old ~ gdSectorName, data = data[which(data$leftEmployer==1),], las=2,
        ylab = "First Job Length (in months)")
title("Transitioned to New Employer")

# looks very different than those who left
boxplot(jobLength_old ~ gdSectorName, data = data[which(data$leftEmployer==0),], las=2, ylab = "First Job Length (in months)")
title("Transitioned Within Employer")

# work life balance
boxplot(ratingWorkLifeBalance ~ gdSectorName, data = data, las=2, ylab = "Work Life Balance Rating")
title("Work Life Balance Rating by Industry")

boxplot(ratingWorkLifeBalance ~ gdSectorName, data = data[which(data$leftEmployer==0),], las=2, ylab = "Work Life Balance Rating")
title("Transitioned Within Employer")

boxplot(ratingWorkLifeBalance ~ gdSectorName, data = data[which(data$leftEmployer==1),], las=2,
        ylab = "Work Life Balance Rating")
title("Transitioned to New Employer")







###################################################################
# EXTRA CODE (UNUSED) #############################################
###################################################################

# #---------------------------#
# # PROPENSITY SCORE MATCHING #
# #---------------------------#
# 
# library(MatchIt)
# df$stayed <- ifelse(df$leftEmployer==0,1,0)
# set.seed(1)
# matched <- matchit(stayed ~ jobTitle + metro + gdSectorName, data = df,
#                    method = "nearest")
# 
# # matching success
# plot(matched, type = "hist")
# m.data <- match.data(matched)
# 
# 
# 
# # basic model: overall rating rating
# fit1_m <- lm(leftEmployer ~ ratingOverall,data = m.data)
# summary(fit1)
# 
# # basic model: subratings
# fit2_m <- lm(leftEmployer ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance
#            + ratingSeniorManagement + ratingBusinessOutlook, data = m.data)
# summary(fit2_m)
# 
# # complex model: job properties, overall rating
# fit3_m <- lm(leftEmployer ~ ratingOverall + yearFounded + empSize_bucket
#            + jobLength_old + startYear + salaryIncrease + logSalary
#            + newMinusOldSalary, data = m.data)
# summary(fit3_m)
# 
# # complex model: job properties, subratings
# fit4_m <- lm(leftEmployer ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance
#            + ratingSeniorManagement  + ratingBusinessOutlook + yearFounded + empSize_bucket + jobLength_old
#            + startYear + salaryIncrease + logSalary + newMinusOldSalary, data = m.data)
# summary(fit4_m)
# 
# 
# # all model: overall rating
# fit5_m <- lm(leftEmployer ~ ratingOverall + yearFounded + empSize_bucket
#            + jobLength_old + startYear + metro + jobTitle + gdSectorName
#            + salaryIncrease + logSalary + newMinusOldSalary, data = m.data)
# summary(fit5_m)
# 
# # all model: subratings
# fit6_m <- lm(leftEmployer ~ ratingCeo  + ratingBusinessOutlook
#            + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance
#            + ratingSeniorManagement + yearFounded + empSize_bucket +
#              jobLength_old + startYear + metro + jobTitle + gdSectorName + salaryIncrease
#            + logSalary + newMinusOldSalary, data = m.data)
# summary(fit6_m)
# 
# 
# 
# ### Output ### -- I would use model 6 here. Adding metro, GOC and sector on a small data set results in severe over fitting.
# 
# vc_1_m <- vcovHC(fit1_m, "HC1")
# vc_2_m <- vcovHC(fit2_m, "HC1")
# vc_3_m <- vcovHC(fit3_m, "HC1")
# vc_4_m <- vcovHC(fit4_m, "HC1")
# vc_5_m <- vcovHC(fit5_m, "HC1")
# vc_6_m <- vcovHC(fit6_m, "HC1")
# 
# se_1_m <- sqrt(diag(vc_1_m))
# se_2_m <- sqrt(diag(vc_2_m))
# se_3_m <- sqrt(diag(vc_3_m))
# se_4_m <- sqrt(diag(vc_4_m))
# se_5_m <- sqrt(diag(vc_5_m))
# se_6_m <- sqrt(diag(vc_6_m))
# 
# #----------#
# #  OUTPUT  #
# #----------#
# stargazer(fit1_m, fit2_m, fit3_m, fit4_m, fit5_m, fit6_m, 
#           se = list(se_1_m, se_2_m, se_3_m, se_4_m, se_5_m, se_6_m),
#           title = "Linear Probability Model Results: Matched Data",
#           type = 'html',
#           omit = c("jobTitle", "gdSector", "metro"),
#           out = "/Users/morgan.smart/Desktop/git/glassdooreconomicresearch/Research/Likelihood_of_Exit/outTable_ADC_matched.html")

# ### Important Feature Plots
# # Job Length
# ggplot(df, aes(x = jobLength_old))+
#   geom_density(aes(fill =factor(df$leftEmployer),colour =factor(df$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density")+
#   xlab("Prior Job Length (in months)")
# # Salary Increase
# ggplot(df_all, aes(newMinusOldSalary))+
#   geom_density(aes(fill =factor(df_all$leftEmployer),colour =factor(df_all$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density") +
#   xlab("Difference in New Job and Prior Job Avg. Salary")
# # logSalary
# ggplot(df_all, aes(x = avgSalary))+
#   geom_density(aes(fill =factor(df_all$leftEmployer),colour =factor(df_all$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density")+
#   xlab("Prior Job Average Salary")
# # Rating Overall
# ggplot(df, aes(ratingOverall))+
#   geom_density(aes(fill =factor(df$leftEmployer),colour =factor(df$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density")+
#   xlab("Prior Employer's Overall Rating")
# # Rating Culture and Valaues
# ggplot(df, aes(ratingCultureAndValues))+
#   geom_density(aes(fill =factor(df$leftEmployer),colour =factor(df$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density")+
#   xlab("Prior Employer's Culture & Values Rating")
# # Rating Career Opps
# ggplot(df, aes(ratingCareerOpportunities))+
#   geom_density(aes(fill =factor(df$leftEmployer),colour =factor(df$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Density")+
#   xlab("Prior Employer's Career Opportunities Rating")
# # start year
# ggplot(df, aes(startYear))+
#   geom_bar(aes(fill =factor(df$leftEmployer),colour =factor(df$leftEmployer)), alpha=0.3)+
#   scale_fill_manual(name='Transition Type',values= c('#2c84cc','#7cb228'), labels=c('Within Employer','New Employer')) +
#   scale_colour_manual(values= c('#2c84cc','#7cb228'), guide=FALSE)+ylab("Count")+
#   xlab("Prior Job Start Year")

# -----------------------------------------------------------------------------
# #############################
# #### STAYED AT EMPLOYER #####
# #############################
# 
# df_s <- df[,-21]
# 
# 
# #---------------#
# #    MODELS     #
# #---------------#
# 
# # basic model: overall rating rating
# fit1_s <- lm(jobLength_old ~ ratingOverall,data = df_s)
# summary(fit1_s)
# 
# # basic model: subratings
# fit2_s <- lm(jobLength_old ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues
#              + ratingWorkLifeBalance + ratingBusinessOutlook
#              + ratingSeniorManagement, data = df_s)
# summary(fit2_s)
# 
# # complex model: job properties, overall rating
# fit3_s <- lm(jobLength_old ~ ratingOverall + yearFounded + empSize_bucket
#              + startYear + salaryIncrease + logSalary + newMinusOldSalary, data = df_s)
# summary(fit3_s)
# 
# # complex model: job properties, subratings
# fit4_s <- lm(jobLength_old ~ ratingCeo + ratingCareerOpportunities + ratingCultureAndValues
#              + ratingWorkLifeBalance + ratingBusinessOutlook
#              + ratingSeniorManagement + yearFounded + empSize_bucket
#              + startYear + salaryIncrease + logSalary + newMinusOldSalary, data = df_s)
# summary(fit4_s)
# 
# 
# # all model: overall rating
# fit5_s <- lm(jobLength_old ~ ratingOverall + yearFounded + empSize_bucket
#              + startYear + metro + jobTitle + gdSectorName + salaryIncrease
#              + logSalary + newMinusOldSalary, data = df_s)
# summary(fit5_s)
# 
# # all model: subratings
# fit6_s <- lm(jobLength_old ~ ratingCeo + ratingBusinessOutlook
#              + ratingCareerOpportunities + ratingCultureAndValues + ratingWorkLifeBalance
#              + ratingSeniorManagement + yearFounded + empSize_bucket +
#                startYear + metro + jobTitle + gdSectorName + salaryIncrease + logSalary
#              + newMinusOldSalary, data = df_s)
# summary(fit6_s)
# 
# # all data
# stargazer(fit1_s, fit2_s, fit3_s, fit4_s, fit5_s, fit6_s,
#           title = "Linear Model Results",
#           type = 'html',
#           omit = c("jobTitle", "gdSector", "metro"),
#           out = "/Users/morgan.smart/Desktop/git/glassdooreconomicresearch/Research/Likelihood_of_Exit/outTable_ADC_robust_s.html")
# 
# # main model
# stargazer(fit5_s,
#           title = "Linear Model Results",
#           type = 'html',
#           omit = c("jobTitle", "gdSector", "metro"),
#           out = "/Users/morgan.smart/Desktop/git/glassdooreconomicresearch/Research/Likelihood_of_Exit/mainModel_s.html")
