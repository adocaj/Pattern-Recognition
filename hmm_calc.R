
###########################################################
#   Author:         Andris Docaj
#
#   
# 
#   Course:         CS5341 section 001
#
#   File Name:      hmm_calc.R
#
#   Specification:  First, we randomly choose a session from 2 users
#                   (let's say user 1 and 2) to train our HMM with.
#                   Then we randomly pick two other different sessions
#                   from these same two users to test our HMM model.
#                   Finally, we also test our trained HMM on data from
#                   sessions of two different users, let's say users 3 and 4.
###########################################################


# here we train and test the hmm using different datasets from
# the same two users


# read the data in
train <- read.csv("train_data.csv", stringsAsFactors = FALSE)
test <- read.csv("test_same_users.csv", stringsAsFactors = FALSE)


# assign start probability and states
# N is a number state, while NN is a no-number state
states <- c("N", "NN")
start_prob <- c(0.35, 0.65)

# the symbols of the observables, S stands for short time period,
# while L stands for a long time period.
elems <- c("S", "L")

# create the transition matrix
n_row <- c(0.149, 1 - 0.149)
nn_row <- c(0.549, 1 - 0.549)

transit_mat <- matrix(c(n_row, nn_row), 2, byrow = TRUE)

# create the emission matrix
n_row_obs <- c(0.149, 1 - 0.149)
nn_row_obs <- c(0.649, 1 - 0.649)

emiss_mat <- matrix(c(n_row_obs, nn_row_obs), 2, byrow = TRUE)

# create the observation sequence to train the hmm
obs <- train$time

# here we train the hmm
library(HMM)

hmm_model <- initHMM(States = states, Symbols = elems,
startProbs = start_prob, transProbs = transit_mat,
emissionProbs = emiss_mat)

# we apply the baum-welch function
baum_welch <- baumWelch(hmm = hmm_model, observation = obs,
maxIterations = 5000, pseudoCount = 0.4)

# observation sequence used to score the model
obs_test <- test$time


# calculate the beta matrix from the trained model
beta_mat <- backward(baum_welch$hmm, observation = obs_test)

# calculate the alpha matrix from the trained model
alpha_mat <- forward(baum_welch$hmm, observation = obs_test)

# determine the column sums of the alpha matrix
alpha_mat.col_sums <- colSums(exp(alpha_mat))

# we calculate the gamma matrix
gamma_mat <- (exp(alpha_mat) * exp(beta_mat)) /
alpha_mat.col_sums[length(alpha_mat.col_sums)]


# create a prediction column all initialized to zero
test$pred <- 0


# if the probability of an N state is higher score it as a 1
start <- 1
end <- ncol(gamma_mat)
for (i in start:end) {
    if (gamma_mat[1,i] > gamma_mat[2,i]) {
        test$pred[i] <- 1
    }
}


# we now plot the prediction using a ROC curve
library(ggplot2)
library(pROC)

roc_obj <- roc(test$pred, test$actual)
area <- round(auc(test$pred, test$actual), 4)

# create roc plot
ggroc(roc_obj, color = 'steelblue', size = 2, legacy.axes = TRUE)+
ggtitle(paste0("ROC curve", " (AUC = ", area, ")"))+
xlab("FPR")+
ylab("TPR")
ggsave('plot.png', path = '~/Desktop/courses/cs5341', dpi = 300)

##############################################################
# Now we do the analysis for different users

test <- read.csv("test_diff_users.csv", stringsAsFactors = FALSE)

# observation sequence used to score the model
obs_test <- test$time

# calculate the beta matrix from the trained model
beta_mat <- backward(baum_welch$hmm, observation = obs_test)

# calculate the alpha matrix from the trained model
alpha_mat <- forward(baum_welch$hmm, observation = obs_test)

# determine the column sums of the alpha matrix
alpha_mat.col_sums <- colSums(exp(alpha_mat))

# we calculate the gamma matrix
gamma_mat <- (exp(alpha_mat) * exp(beta_mat)) /
alpha_mat.col_sums[length(alpha_mat.col_sums)]

# create a prediction column all initialized to zero
test$pred <- 0

# if the probability for state N is larger, score the value as 1
start <- 1
end <- ncol(gamma_mat)
for (i in start:end) {
    if (gamma_mat[1,i] > gamma_mat[2,i]) {
        test$pred[i] <- 1
    }
}


# we now plot the prediciton using a ROCR curve
roc_obj <- roc(test$pred, test$actual)
area <- round(auc(test$pred, test$actual), 4)

# create roc plot
ggroc(roc_obj, color = 'steelblue', size = 2, legacy.axes = TRUE)+
ggtitle(paste0("ROC curve", " (AUC = ", area, ")"))+
xlab("FPR")+
ylab("TPR")
ggsave('plot.png', path = '~/Desktop/courses/cs5341', dpi = 300)

