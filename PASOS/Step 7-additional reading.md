#LOGISTIC REGRESSION

Suppose we want to project the outcomes of coin flips.
We all know that for a fair coin, the expected value is 50 percent heads and 50 percent tails.

What if we had instead an unfair coin, like with a bend in it?
Now let's say we want to generalize coin-flip prediction to all coins, fair and unfair, big and small, heavy and light, et cetera.

What features could we use to predict whether a flip would be heads or tails?
Perhaps we could use the angle of the bend because it disabilities X percent of mass in
the other dimension and/or creates a difference in rotation due to air resistance or center of mass.
The mass of the coin might also be a good feature to know as well as size, properties such as diameter, thickness, et cetera.
We could use some feature engineering on this to get the volume of the coin and furthermore the density.
Maybe the type of material or materials the coin is composed of would be useful information.
These features would be pretty easy to measure.
However, they're only one side of the coin, pun intended.
The rest comes down to the action of the flip itself, such as how much linear/angular velocity the
coin was given, the angle of launch, the angle of what it lands on, wind speed, et cetera.
These might be a bit harder to measure.

Now that we have all these features, what's the simplest model we could use to project heads or tails?
Linear regression, of course!

What could go wrong with this choice, though?
Our labels are heads or tails, or thought of in another way, heads or not heads,
which you can represent with a one-hot encoding of one for heads and zero for not heads.
But if we use linear regression with the standard mean squared error loss function, our predictions could end up being outside the range of zero and one.
What does it mean if we predict 2.75 for the coin-flip state?
That makes no sense.
A model that minimizes square error is under no constraint to treat as a probability in zero to one, but this is what we need here.
In particular, you could imagine a model that predicts values less than zero or greater than one for some new examples.
This would mean we can't use this model as a probability.
Simple tricks, like capping the predictions at zero or one, would introduce bias, so we need something else.
In particular: a new loss function.
Converting this from linear regression to logistic regression can solve this dilemma.
rom an earlier course of ours, we went through the history of ML and introduced the sigmoid activation function.
Let's take a deeper look into that now.
The sigmoid activation function essentially takes the weighted sum, w transpose x, plus b, from a linear regression, and instead of just outputting that and then calculating
the mean squared error loss, we change the activation function from linear to sigmoid, which takes that as an argument and squashes it smoothly between zero and one.
The input into the sigmoid, normally the output of linear regression, is called the logit.
So we are performing a nonlinear transformation on our linear model.
Notice how the probability asymptotes to zero when the logits go to negative infinity and to one when the logits go to positive infinity.

What does this imply for training?
Unlike mean squared error, the sigmoid never guesses 1.0 or 0.0 probability.
This means that in gradient descent's constant drive to get the loss closer and closer to zero, it will drive
the weights closer and closer to plus or minus infinity in the absence of regularization which can lead to problems.

First, though, how can we interpret the output of a sigmoid?
Is it just some function that's range is zero to one, of which there are many, or is it something more?
The good news is that it is something more.
It is a calibrated probability estimate.
Beyond just the range, the sigmoid function is the cumulative distribution function of the logistic
probability distribution, whose quantile function is the inverse of the logit which models the long odds.
Therefore, mathematically, the outputs of a sigmoid can be considered probabilities.
In this way, we can think of calibration as the fact the outputs are real-world values like probabilities.
This is in contrast to uncalibrated outputs, like an embedding vector, which is internally informative, but the values have no real correlation.
Lots of output activation functions, in fact, an infinite number, could give you a number between zero and one,
but only this sigmoid is proven to be a calibrated estimate of the training data set probability of occurrence.
Using this fact about the sigmoid-activation function, we can cast binary-classification problems into probabilistic problems.
For instance, instead of a model just predicting a yes or a no, such as, "Will
a customer buy an item?" it can now predict the probability that a customer buys an item.
This paired with a threshold can provide a lot more predictive power than just a simple binary answer.
However, regularization is important in logistic regression because driving loss to zero is difficult and dangerous.
First, as gradient descent seeks to minimize cross entropy, it pushes output values closer to one for positive labels and closer to zero for negative labels.
Due to the equation of the sigmoid, the function asymptotes to zero when the logit is negative infinity and to one when the logit is positive infinity.
To get the logits to negative or positive infinity, the manager of the weights is increased and increased, leading to numerical-stability problems, overflows and underflows.
This is dangerous and can ruin our training.
Also near the asymptotes, as you can see from the graph, the sigmoid function becomes flatter and flatter.
This means that the derivative is getting closer and closer to zero.
Since we used the derivative and back propagation to update the weights, it is important for the gradient not to become zero or else training will stop.
This is called saturation, when all activations end up in these plateaus, which leads to a vanishing-gradient problem and makes training difficult.
This is also a potentially useful insight here.
Imagine you assign a unique ID for each example and map each ID to its own feature.
If you use unregularized logistic regression, this will lead to absolute overfitting, as the model tries to drive loss to zero
on all examples and never gets there, the weights for each indicator feature will be driven to positive infinity or negative infinity.
This can happen in practice in high- dimensional data with feature crosses.
Often, there's a huge mass of rare crosses that happens only on one example each.

So how can we protect ourselves from overfitting?
Which of these is important when performing logistic regression?
The correct answer is both A and B. Adding regularization to logistic regression helps keep the model simpler by having smaller parameter weights.
This penalty term added to the loss function makes sure that cross entropy through gradient descent doesn't
keep pushing the weights from closer to closer to plus or minus infinity and causing numerical issues.
Also with now smaller logits, we can now stay in the less-flat portions of the sigmoid
function, making our gradients less closer to zero and thus allowing weight updates and training to continue.
C is incorrect.

Therefore, so is E because regularization does not transform the outputs into calibrated probability estimate.
The great thing about logistic regression is that it already outputs the calibrated probability estimate, since the sigmoid function is accumulated distribution function of the logistic-probability distribution.
This allows us to actually predict probabilities instead of just binary answers, like yes or no, true or false, buy or sell, et cetera.
To counteract overfitting, we often do both regularization and early stopping.
For regularization, model complexity increases with large weights, and so as we tune and start to get
larger and larger weights for rarer and rarer scenarios, we end up increasing the loss, so we stop.
L2 regularization will keep the weight values smaller, and L1 regularization will keep the models sparser by dropping poor features.
To find the optimal L1 and L2 parameter choices during hyper parameter tuning, you are searching for the point in the validation-loss function where you obtain the lowest value.
At that point, any less regularization increases your variants, starts overfitting and hurts generalization, and any more regularization increases your bias, starts underfitting and hurts your generalization.
Early stopping stops training when overfitting begins.
As you train your model, you should evaluate your model on your validation data set every so many steps, epochs, minutes, et cetera.
As training continues, both the training error and the validation error should be decreasing, but at some point the validation error might begin to actually increase.
It is at this point that the models begin to memorize the training data set and lose its ability to generalize
to the validation data set, and most importantly, to the new data that we will eventually want to use this model for.
Using early stopping would stop the model at this point and then back up and use the weights from the previous step before it hit validation-error-inflection point.
Here, the loss is just L(w,D), i.e., no regularization term.
Interestingly, early stopping is an approximate equivalent of L2 regularization and is often used in its place because it is computationally cheaper.
Fortunately, in practice, we always use both explicit regularization, L1 and L2, and also some amount of early stopping regularization.
Even though L1 regularization and early stopping seem a bit redundant, for real-world systems, you may not
quite choose the optimal hyper parameters, and thus early stopping can help fix that choice for you.
It's great that we can obtain a probability from our logistic-regression model.
However, at the end of the day, sometimes users just want a simple decision to be made for them for their real-world problems.

Should the e-mail be sent to the spam folder or not?
Should the loan be approved or not?
Which road should we route the user through?
How can we use a probability estimate to help the tool, used in our model, to make a decision?
We choose a threshold.

A simple threshold of a binary-classification problem would be all probabilities less than or equal to
50 percent should be a no, and all probabilities greater than 50 percent should be a yes.
However, for studying real-world problems, we may want a different split, like 60/40, 20/80, 99/1, et cetera, depending on how we want
our balance of our type one and type two errors, or in other words, our balance of false positives and false negatives.
For binary classification, we will have four possible outcomes: true positives, true negatives, false positives and false negatives.
Combinations of these values can lead to evaluation metrics like precision, which is the number of true positives divided by all positives, and recall,
which is the number of true positives divided by the sum of true positives and false negatives which gives the sensitivity or true-positive rate.
You can tune your choice of threshold to optimize the metric of your choice.
Is there any easy way to help us do this?
A receiver operating characteristic curve, or ROC curve for short, shows how a given model's predictions create different true positive versus false-positive rates when different decision thresholds are used.
As we lower the threshold, we are likely to have more false positives, but we'll also increase the number of true positives we find.
Ideally, a perfect model would have zero false positives and zero false negatives, which plugging that
into the equations would give a true-positive rate of one and a false-positive rate of zero.
To create a curve, we would pick each possible decision threshold and re-evaluate.
Each threshold value creates a single point, but by evaluating many thresholds, eventually a curve is formed.
Fortunately, there's an efficient sorting-based algorithm to do this.
Each model will create a different ROC curve, so how can we use these curves to compare

the relative performance of our models when we don't know exactly what decision threshold we want to use?
We can use the area under the curve as an aggregate measure of performance across all possible classification thresholds.
AUC helps you choose between models when you don't know what decision threshold is going to ultimately used.

It is like asking, "If we pick a random positive and a random negative, what's the probability my model scores them in the correct relative order?"
The nice thing about AUC is that it's scale invariant and classification-threshold invariant.
People like to use it for those reasons.
People sometimes also use AUC for the precision-recall curve, or more recently, precision-recall-gain curves, which just use different combinations of the four prediction outcomes as metrics along the axes.
However, treating this only as an aggregate measure can mask some effects.
For example, a small improvement in AUC might come by doing a better job of ranking
very unlikely negatives as even still yet more unlikely, which is fine, but potentially not materially beneficial.
When we evaluate our logistic-regression models, we need to make sure predictions aren't biased.
When we talk about bias in this sense, we are not talking about the bias term in the model's linear equation.
Instead, we mean there shouldn't be an overall shift in either the positive or negative direction.
A simple way to check the prediction bias is to compare the average value predictions made by
the model over a data set to the average value of the labels in that data set.
If they are not relatively close, then you might have a problem.
Bias is like a canary in a mind, where we can use it as an indicator of something being wrong.
If you have bias, you definitely have a problem, but even zero bias alone does not mean everything in your system is perfect, but it is a great sanity check.
If you have bias, you could have an incomplete feature set, a buggy pipeline, a biased training sample, et cetera.
You can look for bias in slices of data, which could help guide improvements over removing bias from your model.
Let's look at an example of how you can do that.
Here is a calibration plot from the sibyl experiment browser.
You'll notice that this is in a log-log scale, as we're comparing the bucketized log odds predicted to the bucketized log odds observed.
You'll note that things are pretty well calibrated in the moderate range, but the extreme low end is pretty bad.
This can happen when parts of the data space is not well represented or because of noise or because of overly strong regularization.
The bucketing can be done in a couple of ways.
You can bucket by literally breaking up the target predictions, or we can bucket by quantiles.

Why do we need to bucket prediction to make calibration plots when predicting probabilities?
For any given event, the true label is either zero or one.
For example, not clicked or clicked.
But our prediction values will always be a probabilistic guess somewhere in the middle, like 0.1 or 0.33.
For any individual example, we're always off, but if we group enough examples together, we'd like to see that
on average the sum of the true zeroes and ones is about the same as the mean probability we're predicting.

Which of these is important in performing logistic regression?
The correct answer is all of the above.
It is extremely important that our model generalizes so that we have the best predictions on new data, which is the entire reason we created the model to begin with.
To help do this is important that we do not overfit our data.
Therefore, adding penalty terms to the objective function, like with L1 regularization for sparsity and
L2 regularization for keeping model weight small and adding early stopping can help in this regard.
It is also important to choose a tuned threshold for deciding what decisions to make
when your probability estimate outputs to minimize or maximize the business metric as important to you.
If this isn't well defined, then we can use more statistical means, such as calculating the number of true
and false positives and negatives and combining them into different metrics, such as the true and false positive rates.
We can then repeat this process for many different thresholds and then plot the area
under the curve, or AUC, to come up with a relative aggregate measure of model performance.
Lastly, it is important that our predictions are unbiased, and even if there isn't bias, we should be still diligent to make sure our model is performing well.
We begin looking for bias by making sure that the average of the predictions is very close to the average of observations.
A helpful way to find where bias might be hiding is to look at slices of
data and use something like a calibration plot to isolate the problem areas for further refinement.
