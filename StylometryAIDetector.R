# Functions

install.packages('class')
install.packages('caret')

library(class)
library(caret)

#load in a literary corpus. Filedir should be the directory of the function words, which contains one folder for
#each author. The 'featureset' argument denotes the type of features that should be used
loadCorpus <- function(filedir,featureset="functionwords",maxauthors=Inf) {
  authornames <- list.files(filedir)
  booknames <- list()
  features <- list()
  count <- 0
  
  for (i in 1:length(authornames)) {
    #print(i)
    if (count >= maxauthors) {break}
    files <- list.files(sprintf("%s%s/",filedir,authornames[i]))
    if (length(files)==0) {next}
    
    firstbook <- FALSE
    booknames[[i]] <- character()
    for (j in 1:length(files)) {
      path <- sprintf("%s%s/%s",filedir,authornames[i],files[j])
      
      fields <- strsplit(files[j],split=' --- ')[[1]]  
      
      if (sprintf("%s.txt",featureset) == fields[2]) {
        booknames[[i]] <- c(booknames[[i]], fields[1])
        count <- count+1
        M <- as.matrix(read.csv(path,sep=',',header=FALSE))  
        if (firstbook == FALSE) {
          firstbook <- TRUE
          features[[i]] <- M
        } else {
          features[[i]]  <- rbind(features[[i]],M)
        }
        
      }
    }
  }
  return(list(features=features,booknames=booknames,authornames=authornames))
}

myKNN <- function(traindata, testdata, trainlabels, k=1) {
  if (mode(traindata) == 'numeric' && !is.matrix(traindata)) {
    traindata <- matrix(traindata,nrow=1)
  }
  if (mode(testdata) == 'numeric' && !is.matrix(testdata)) {
    testdata <- matrix(testdata,nrow=1)
  }
  
  mus <- apply(traindata,2,mean) 
  sigmas <- apply(traindata,2,sd)
  
  for (i in 1:ncol(traindata)) {
    traindata[,i] <- (traindata[,i] - mus[i])/sigmas[i]
  }
  
  for (i in 1:ncol(testdata)) {
    testdata[,i] <- (testdata[,i]-mus[i])/sigmas[i]
  }
  
  preds <- knn(traindata, testdata, trainlabels, k)
  return(preds)
}

discriminantCorpus <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  #first learn thea model for each aauthor
  for (i in 1:length(traindata)) {
    words <- apply(traindata[[i]],2,sum)
    
    #some words might never occur. This will be a problem since it will mean the theta for this word is 0, which means the likelihood will be 0 if this word occurs in the training set. So, we force each word to occur at leats once
    inds <- which(words==0) 
    if (length(inds) > 0) {words[inds] <- 1}
    thetas <- rbind(thetas, words/sum(words))
  }
  
  #now classify
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      probs <- c(probs, dmultinom(testdata[i,],prob=thetas[j,],log=TRUE))
    }
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}


KNNCorpus <- function(traindata, testdata) {
  train <- NULL
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  trainlabels <- 1:nrow(train)
  myKNN(train, testdata, trainlabels,k=1)
}

randomForestCorpus <- function(traindata, testdata) {
  x <- NULL
  y <- NULL
  for (i in 1:length(traindata)) {
    x <- rbind(x,traindata[[i]])
    y <- c(y,rep(i,nrow(traindata[[i]])))
  }
  
  for (i in 1:nrow(x)) {
    x[i,] <- x[i,]/sum(x[i,])
  }
  
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  mus <- apply(x,2,mean)
  sigmas <- apply(x,2,sd)
  for (j in 1:ncol(x)) {
    x[,j] <- (x[,j] - mus[j])/sigmas[j]
    testdata[,j] <- (testdata[,j] - mus[j])/sigmas[j]
  }
  
  y <- as.factor(y)
  rf <- randomForest(x,y)
  
  preds <- numeric(nrow(testdata))
  for (i in 1:nrow(testdata)) {
    preds[i] <- predict(rf,testdata[i,])
  }
  return(preds)
}
# Random Forest

install.packages("randomForest")
library(randomForest)
library(caret)
features_combined <- do.call(rbind, training$features) # Combine ChatGPT and Human rows
#label vector for the combined data
labels <- rep(training$authornames, each = nrow(training$features[[1]]))
# Combine into a data frame for easier manipulation
training_data <- data.frame(features_combined, Author = labels)
X <- training_data[, -ncol(training_data)] # All columns except 'Author'
y <- as.factor(training_data$Author) # The 'Author' column is the target
# training the Random Forest model
rf_model <- randomForest(X, y, ntree =100, importance = TRUE)
# print a summary of the model
print(rf_model)
rf_model
importance_values <- importance(rf_model)
importance_df <- as.data.frame(importance_values)
importance_df$Function_Word <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]
importance_df_GPT <- importance_df[order(-importance_df$GPTAll), ]
importance_df_Human <- importance_df[order(-importance_df$HumanAll), ]
importance_df_Gini <- importance_df[order(-importance_df$MeanDecreaseGini), ]
# Display top features
head(importance_df, 10)
head(importance_df_Human, 10)
head(importance_df_Gini, 10)
funcwords <- c("a", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "do", "down", "even", "every", "for", "from", "had", "has", "have", "her", "his", "if", "in", "into", "is", "it", "its", "may", "more", "must", "my", "no", "not", "now", "of", "on", "one", "only", "or", "our", "shall", "should", "so", "some", "such", "than", "that", "the", "their", "then", "there", "things", "this", "to", "up", "upon", "was", "were", "what", "when", "which", "who", "will", "with", "would", "your")
funcwords[58]
funcwords[52]
funcwords[40]
funcwords[54]
importance_df
funcwords[27]
funcwords[30]
funcwords[9]
funcwords[29]
funcwords[2]
funcwords[1]
funcwords[12]
funcwords[2]
funcwords[34]
funcwords[49]
rf_model$err.rate[rf_model$ntree, 1]
importance_df$Mapped_Function_Word <- funcwords[as.numeric(sub("V", "", importance_df$Function_Word))]



plotdata200 <- list()
plotdataacc200 <- list()
for (k in 0:19) {
  featuresGPT <- reducewords(TrainingNoMin$features[[1]],200 - 10*k)
  features <- reducewords(TrainingNoMin$features[[2]],200 - 10*k)
  reducedfeatures <- list(featuresGPT, features)
  features_combined <- do.call(rbind, reducedfeatures) # Combine ChatGPT and Human rows
  #label vector for the combined data
  labels <- rep(training$authornames)
  # Combine into a data frame for easier manipulation
  training_data <- data.frame(features_combined, Author = labels)
  X <- training_data[, -ncol(training_data)] # All columns except 'Author'
  y <- as.factor(training_data$Author) # The 'Author' column is the target
  # training the Random Forest model
  rf_model <- randomForest(X, y, ntree =100, importance = TRUE)
  accuracy <- sum(predictions==truth)/length(truth)
  plotdata200 <- append(plotdata200, 200 - 10 * k)
  plotdataacc200 <- append(plotdataacc200, accuracy)
  print(k)
}




# Step 1: Load the data
# Assuming importance_df is already loaded

# Step 2: Remove non-numerical columns (if any)
importance_numeric <- head(importance_df[, -ncol(importance_df)],10) # Remove 'Function_Word' (last column in example)

# Step 3: Standardize the data
importance_scaled <- scale(importance_numeric)

# Step 4: Perform PCA
pca_result <- prcomp(importance_scaled, center = TRUE, scale. = TRUE)

# Step 5: View PCA summary
summary(pca_result)

# Step 6: Visualize PCA (optional)
# Variance explained by each principal component
screeplot(pca_result, type = "lines", main = "Scree Plot")
biplot(pca_result, main = "PCA Biplot")

library(ggplot2)
ggplot(importance_df, aes(x = reorder(Mapped_Function_Word, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance (Mean Decrease in Accuracy)", x = "Features", y = "Importance")
ggplot(importance_df, aes(x = reorder(Mapped_Function_Word, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance (Mean Decrease in Gini)", x = "Features", y = "Importance")


# Split the data into training and testing sets
train_indices <- sample(1:nrow(training_data), size = 0.5 * nrow(training_data))  # 80% for training
train_data <- training_data[train_indices, ]
test_data <- training_data[-train_indices, ]

# Define features and target for the training set
X_train <- train_data[, -ncol(train_data)]
y_train <- as.factor(train_data$Author)

# Define features and target for the testing set
X_test <- test_data[, -ncol(test_data)]
y_test <- as.factor(test_data$Author)
rf_model <- randomForest(X_train, y_train, ntree = 100, importance = TRUE)

# Print a summary of the model
print(rf_model)
y_pred <- predict(rf_model, X_test)
confusionMatrix(as.factor(y_pred), as.factor(y_test))
accuracy <- sum(y_pred == y_test) / length(y_test)
print(paste("Test Accuracy:", accuracy))


# Random Forest Function Words

features_combined_no71 <- features_combined[, -71]
labels <- rep(training$authornames, each = nrow(training$features[[1]]))
# Combine into a data frame for easier manipulation
training_data <- data.frame(features_reduced_FW, Author = labels)
X <- training_data[, -ncol(training_data)] # All columns except 'Author'
y <- as.factor(training_data$Author) # The 'Author' column is the target
# training the Random Forest model
rf_model <- randomForest(X, y, ntree =100, importance = TRUE)
# print a summary of the model
print(rf_model)
rf_model
importance_values_no71 <- importance(rf_model)
importance_df_no71 <- as.data.frame(importance_values_no71)
importance_df_no71$Function_Word <- rownames(importance_df_no71)
importance_df_no71 <- importance_df_no71[order(-importance_df_no71$MeanDecreaseAccuracy), ]
head(importance_df_no71)
features_reduced_FW <- features_combined[,-c(27, 52, 58, 71)]
labels <- rep(training$authornames, each = nrow(training$features[[1]]))
# Combine into a data frame for easier manipulation
training_data <- data.frame(features_reduced_FW, Author = labels)
X <- training_data[, -ncol(training_data)] # All columns except 'Author'
y <- as.factor(training_data$Author) # The 'Author' column is the target
# training the Random Forest model
rf_model_reduced <- randomForest(X, y, ntree =100, importance = TRUE)
# print a summary of the model
print(rf_model_reduced)
rf_model_reduced


train_indices <- sample(1:nrow(training_data), size = 0.5 * nrow(training_data))  # 60% for training
train_data <- training_data[train_indices, ]
test_data <- training_data[-train_indices, ]

# Define features and target for the training set
X_train <- train_data[, -ncol(train_data)]
y_train <- as.factor(train_data$Author)

# Define features and target for the testing set
X_test <- test_data[, -ncol(test_data)]
y_test <- as.factor(test_data$Author)
rf_model <- randomForest(X_train, y_train, ntree = 100, importance = TRUE)
print(rf_model)

# Print a summary of the model



print(rf_model)
y_pred <- predict(rf_model, X_test)
accuracy <- sum(y_pred == y_test) / length(y_test)
print(paste("Test Accuracy:", accuracy))

importance_values_reduced <- importance(rf_model)
importance_df_reduced <- as.data.frame(importance_values_reduced)
importance_df_reduced$Function_Word <- rownames(importance_df_reduced)
importance_df_reduced <- importance_df_reduced[order(-importance_df_reduced$MeanDecreaseAccuracy), ]
head(importance_df_reduced)
funcwords[5]
funcwords[9]
funcwords[12]
funcwords[55]
funcwords[70]
funcwords[c(9,1,40,12,5,30)]
importance_df_reduced$Mapped_Function_Word <- funcwords[as.numeric(sub("V", "", importance_df_reduced$Function_Word))]
importance_df_reduced
ggplot(importance_df_reduced, aes(x = reorder(Mapped_Function_Word, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance (Mean Decrease in Accuracy)", x = "Features", y = "Importance")
ggplot(importance_df_reduced, aes(x = reorder(Mapped_Function_Word, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance (Mean Decrease in Gini)", x = "Features", y = "Importance")

# Random Forest Length Analysis

library(randomForest)
features_combined <- do.call(rbind, training$features)
labels <- rep(training$authornames, each = nrow(training$features[[1]]))
# Combine into a data frame for easier manipulation
accuracy <- list()
for (k in 1:19) {
  featuresGPT <- reducewords(TrainingNoMin$features[[1]],200 - 10*k)
  features <- reducewords(TrainingNoMin$features[[2]],200 - 10*k)
  reducedfeatures <- list(featuresGPT, features)
  features_combined <- do.call(rbind, reducedfeatures)
  training_data <- data.frame(features_combined, Author = labels[-1955])
  labels <- rep(training$authornames, each = nrow(training$features[[1]]))
  train_indices <- sample(1:nrow(training_data), size = 0.8 * nrow(training_data))  # 80% for training
  train_data <- training_data[train_indices, ]
  test_data <- training_data[-train_indices, ]
  
  #define features and target for the training set
  X_train <- train_data[, -ncol(train_data)]
  y_train <- as.factor(train_data$Author)
  
  #define features and target for the testing set
  X_test <- test_data[, -ncol(test_data)]
  y_test <- as.factor(test_data$Author)
  rf_model <- randomForest(X_train, y_train, ntree = 100, importance = TRUE)
  print(rf_model)
}
y_axis_data_80 <- c(98.88, 98.65, 98.62, 98.56, 98.39, 98.25, 98.36, 97.47, 97.55, 97.81, 97.18, 97.27, 96.09, 95.97, 95.08, 94.62, 93.24, 92.87, 91.69, 91.94, 91.48, 90.56, 89.93, 89.53, 88.23, 88.2, 86.08, 85.96, 85.41, 83.37, 82.59, 79.95, 77.91, 74.25, 70.94, 63.72)
y_axis_data_50 <- c(0.9843, 0.9848, 0.9876, 0.9853, 0.9825, 0.9797, 0.9761, 0.977, 0.9761, 0.9678, 0.9696, 0.9599, 0.9553, 0.9351, 0.942, 0.9203, 0.9203, 0.9199, 0.9153, 0.9033, 0.9047, 0.8923, 0.878, 0.8748, 0.8692, 0.8591, 0.861, 0.8444, 0.8269, 0.8089, 0.7753, 0.762, 0.7164, 0.6911, 0.6616)
x_axis_data <- c(1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
y_axis_data_100 <- c(0.9889528, 0.9864212, 0.9880322, 0.9852704, 0.9857307, 0.9836594, 0.9822785, 0.9806674, 0.9792865, 0.9788262, 0.9774453, 0.972382, 0.9643268, 0.9668585, 0.950748, 0.941542, 0.9360184, 0.9173763, 0.9194476, 0.9196778, 0.9132336, 0.9120829, 0.9017261, 0.8980437, 0.8966628, 0.8766398, 0.8642117, 0.8559264, 0.8469505, 0.8347526, 0.8174914, 0.8048331, 0.7689298, 0.7314154, 0.7026467, 0.6388953)
plot(x_axis_data, y_axis_data)
for (k in 0:19) {
  featuresGPT <- reducewords(TrainingNoMin$features[[1]],200 - 10*k)
  features <- reducewords(TrainingNoMin$features[[2]],200 - 10*k)
  reducedfeatures <- list(featuresGPT, features)
  features_combined <- do.call(rbind, reducedfeatures)}


accuracy
training_data <- data.frame(features_combined, Author = labels[-1955])
X <- training_data[, -ncol(training_data)] # All columns except 'Author'
y <- as.factor(training_data$Author) # The 'Author' column is the target
# training the Random Forest model
rf_model <- randomForest(X, y, ntree =100, importance = TRUE)
# print a summary of the model
print(rf_model)
error_rate_1000 <- rf_model$err.rate[rf_model$ntree, 1]
error_rate_950 <- rf_model$err.rate[rf_model$ntree, 1]


# Multidimensional Scaling

library(dendextend)
# M <- use loadcorpus function to load in all function words
# HumanTraining <- use loadcorpus function to load in Human function words
# ChatGPTTraining <- use loadcorpus function to load in GPT function words
# training <- use loadcorpus function to load in training function words
x <- NULL
for (i in 1:length(M$features)) {
  x <- rbind(x, apply(M$features[[i]],2,sum))
}
for (i in 1:nrow(x)) {
  x[i,] <- x[i,] / sum(x[i,])
}
for (j in 1:ncol(x)) {
  x[,j] <- (x[,j] - mean(x[,j]))/sd(x[,j])
}

x


d <- dist(x) #create a distance matrix
pts <- cmdscale(d)
plot(pts,type='n')
text(pts[,1],pts[,2],cex=0.8)

d <- dist(x, method = "euclidean") # Distance matrix: Euclidean on standarised variables
hc <- hclust(d, method="ward.D") # Ward's clustering method
plot(hc,labels=M$authornames) # Dendrogram
rect.hclust(hc , h =20)
hc

predictions <- NULL
KNNpredictions <- NULL
truth <- NULL
features <- training$features
for (i in 1:length(features)) {
  for (j in 1:nrow(features[[i]])) {
    testdata <- matrix(features[[i]][j,],nrow=1)
    traindata <- features
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    pred <- discriminantCorpus(traindata, testdata)
    predictions <- c(predictions, pred)
    #pred <- KNNCorpus(traindata, testdata)
    #KNNpredictions <- c(KNNpredictions, pred)
    truth <- c(truth, i)
  }
}
sum(predictions==truth)/length(truth)
sum(KNNpredictions==truth)/length(truth)
confusionMatrix(as.factor(predictions), as.factor(truth))
misclassified <- which(predictions != y)
predictions <- factor(predictions, levels = c(1, 2), labels = c("GPTAll", "HumanAll"))
predictions[-1955]
# Compare predictions to actual labels (y)
misclassified <- which(predictions[-4300] != y)
y
print(misclassified)
print(training$features[misclassified])
# TrainingGroup1 <- use loadcorpus function to load in corpus
sum(training$features[[2]][2173,])

training$features[[1]][misclassified[1:100],]
TrainingGroup3CM
print(TrainingGroup1Discriminant)
variable_list <- lapply(1:8, function(x) get(paste0("TrainingGroup", x, "KNN")))

# Print the list of values
print(variable_list)


ggplot(df2) +
  geom_point(aes(x = X1, y = X2, colour = "blue")) +
  geom_text(aes(x = X1, y = X2, label=row.names(df2), colour = "blue"), hjust = 0, nudge_x = 0.2) +
  geom_point(aes(x = X11, y = X21)) +
  geom_text(aes(x = X11, y = X21, label=row.names(df2)), hjust = 0, nudge_x = 0.2) +
  xlim(c(-14,10.5)) +
  ylim(c(-10,9)) +
  labs(title = "MDS Plot of Categories", x = "MDS Dimension 1", y = "MDS Dimension 2")+
  theme_bw()


# Length Plot

library(ggplot2)

data_x_axis <- c(1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,190,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10)
y_axis_data_RF <- c(98.88, 98.65, 98.62, 98.56, 98.39, 98.25, 98.36, 97.47, 97.55, 97.81, 97.18, 97.27, 96.09, 95.97, 95.08, 94.62, 93.24, 92.87, 91.69, 91.94, 91.48, 90.56, 89.93, 89.53, 88.23, 88.2, 86.08, 85.96, 85.41, 83.37, 82.59, 79.95, 77.91, 74.25, 70.94, 63.72)
data_y_axis <- c(0.9622555, 0.9622555, 0.9597238, 0.961565, 0.9604143, 0.9592635, 0.9634062, 0.9578826, 0.9583429, 0.9590334, 0.9548907, 0.9546605, 0.9537399, 0.9498274, 0.9429229, 0.9373993, 0.9385501, 0.9302647, 0.9268124, 0.9157652, 0.9162255, 0.912313, 0.9166858, 0.8973533, 0.9012658, 0.8966628, 0.8817031, 0.8800921, 0.8711162, 0.8626007, 0.8349827, 0.8299194, 0.7942463, 0.7631761, 0.7275029, 0.6644419)
data_y_axis <- data_y_axis * 100
df1 <-data.frame(now = data_x_axis, accuracy = data_y_axis)
df2 <-data.frame(now = x_axis_data, accuracy = y_axis_data_RF)

ggplot() +
  geom_line(data = df1, aes(x = now, y = accuracy, color = "red")) +
  geom_point(data = df1, aes(x = now, y = accuracy)) +
  geom_line(data = df2, aes(x = now, y = accuracy, color = "blue")) +
  geom_point(data = df2, aes(x = now, y = accuracy)) +
  scale_color_manual(labels = c("Random Forest", "Discriminant Analysis"), values = c("blue", "red")) +
  labs(x = "Number of Words", y = "Accuracy (%)", color = "Stylometric Methods") +
  theme_minimal() +
  scale_y_continuous(limits = c(63, 100), breaks = seq(60, 100, 5)) +
  theme(axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.position=c(0.8,0.2))


# Length of Essays

sums_list <- vector("list", 2173)

# Loop over each row from 1 to 2173
for (i in 1:2173) {
  # Calculate the sum of the ith row and store it in the list
  sums_list[[i]] <- sum(training$features[[2]][i, ])
}

#print the list of sums
print(sums_list)
min_sum <- min(unlist(sums_list))
max_sum <- max(unlist(sums_list))
print(min_sum)
print(max_sum)
sumarray <- unlist(sums_list)
count_under_1000 <- (unlist(sums_list) < 1000)

print(count_under_1000)


min_index <- which.min(unlist(sums_list))

print(min_index)
training[2]
TrainingNoMin$features[[2]]
TrainingNoMin[[2]] <- training[[2]][-1955]
training[[2]]
TrainingNoMin$features[[2]][1955,]
TrainingNoMin <- training
training$booknames[[2]][1955]
#remove the 1955th element
TrainingNoMin$features[[2]] <- TrainingNoMin$features[[2]][-1955, ]
TrainingNoMin$booknames[[2]] <- TrainingNoMin$booknames[[2]][-1955]
training$features[[2]] <- training$features[[2]][-1955, ]
training$booknames[[2]] <- training$booknames[[2]][-1955]
training$authornames[[2]] <- training$authornames[[2]][-1955]
TrainingNoMin1000 <- reducewords(TrainingNoMin$features[[2]], 1000)



# Groups Analysis

predictions <- NULL
KNNpredictions <- NULL
truth <- NULL
features <- training$features
for (i in 1:length(features)) {
  for (j in 1:nrow(features[[i]])) {
    testdata <- matrix(features[[i]][j,],nrow=1)
    traindata <- features
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    pred <- randomForestCorpus(traindata, testdata)
    forestpredictions <- c(predictions, y)
    truth <- c(truth, i)
  }
}
TrainingGroupRF <- sum(forestpredictions==truth)/length(truth)
TrainingGroup9KNN <- sum(KNNpredictions==truth)/length(truth)
TrainingGroup9CM <- confusionMatrix(as.factor(predictions), as.factor(truth))


# Length Analysis

TrainingNoMin1000 <- reducewords(TrainingNoMin$features[[2]], 1000)
TrainingNoMin1000GPT <- reducewords(TrainingNoMin$features[[1]], 1000)
TrainingNoMin950 <- reducewords(TrainingNoMin$features[[2]], 950)
TrainingNoMin950GPT <- reducewords(TrainingNoMin$features[[1]], 950)
TrainingNoMin100 <- reducewords(TrainingNoMin$features[[2]], 100)
TrainingNoMin100GPT <- reducewords(TrainingNoMin$features[[1]], 100)
TrainingNoMin50 <- reducewords(TrainingNoMin$features[[2]], 50)
TrainingNoMin50GPT <- reducewords(TrainingNoMin$features[[1]], 50)
TrainingNoMin10 <- reducewords(TrainingNoMin$features[[2]], 10)
TrainingNoMin10GPT <- reducewords(TrainingNoMin$features[[1]], 10)


reducedfeatures1000 <- list(TrainingNoMin1000GPT, TrainingNoMin1000)
reducedfeatures950 <- list(TrainingNoMin950GPT, TrainingNoMin950)
reducedfeatures100 <- list(TrainingNoMin100GPT, TrainingNoMin100)
reducedfeatures50 <- list(TrainingNoMin50GPT, TrainingNoMin50)
reducedfeatures10 <- list(TrainingNoMin10GPT, TrainingNoMin10)

plotdata200 <- list()
plotdataacc200 <- list()
for (k in 0:19) {
  featuresGPT <- reducewords(TrainingNoMin$features[[1]],200 - 10*k)
  features <- reducewords(TrainingNoMin$features[[2]],200 - 10*k)
  reducedfeatures <- list(featuresGPT, features)
  predictions <- NULL
  truth <- NULL
  for (i in 1:length(reducedfeatures)) {
    for (j in 1:nrow(reducedfeatures[[i]])) {
      testdata <- matrix(reducedfeatures[[i]][j,],nrow=1)
      traindata <- reducedfeatures
      traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
      pred <- discriminantCorpus(traindata, testdata)
      predictions <- c(predictions, pred)
      truth <- c(truth, i)
    }
  }
  accuracy <- sum(predictions==truth)/length(truth)
  plotdata200 <- append(plotdata200, 200 - 10 * k)
  plotdataacc200 <- append(plotdataacc200, accuracy)
  print(k)
}
plot(plotdata, plotdataacc)
plot(plotdata200, plotdataacc200)
plotdatatotal <- append(plotdata200, plotdata)
plotdatatotalacc <- append(plotdataacc200, plotdataacc)
plot(plotdatatotal, plotdatatotalacc)
print(plotdatatotal)
predictions <- NULL
truth <- NULL
features <- reducedfeatures10
for (i in 1:length(features)) {
  for (j in 1:nrow(features[[i]])) {
    testdata <- matrix(features[[i]][j,],nrow=1)
    traindata <- features
    traindata[[i]] <- traindata[[i]][-j,,drop=FALSE]
    pred <- discriminantCorpus(traindata, testdata)
    predictions <- c(predictions, pred)
    truth <- c(truth, i)
  }
}
sum(predictions==truth)/length(truth)
confusionMatrix(as.factor(predictions), as.factor(truth))
plot(data_x_axis, data_y_axis)
