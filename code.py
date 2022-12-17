sc.setLogLevel("ERROR")
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bucket = 's3a://amazon-reviews-pds/tsv/'
file_list = ['s3a://amazon-reviews-pds/tsv/amazon_reviews_us_Home_Entertainment_v1_00.tsv.gz',
             's3a://amazon-reviews-pds/tsv/amazon_reviews_us_Home_v1_00.tsv.gz',
             's3a://amazon-reviews-pds/tsv/amazon_reviews_us_Home_Improvement_v1_00.tsv.gz']


sdf = spark.read.csv(file_list, sep='\t', header=True, inferSchema=True)

sdf = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)

sdf = sdf.sample(.25)

sdf = sdf.na.drop(subset=["star_rating", "review_body", "review_date", "vine", "product_category", "verified_purchase"])

sdf = sdf.filter(sdf.product_category.isin(['Home','Home Improvement','Home Entertainment']))

sdf = sdf.withColumn('review_body_wordcount', size(split(col('review_body'), ' ')))

sdf = sdf.withColumn("label", when(col("star_rating") > 3, 1.0).otherwise(0.0))

sdf = sdf.withColumn("total_votes",sdf.total_votes.cast(DoubleType()))
sdf = sdf.withColumn("review_body_wordcount",sdf.review_body_wordcount.cast(DoubleType()))

sdf = sdf.drop("review_body", "review_headline", "marketplace", "customer_id", "review_id", "product_id", "product_parent", "product_title")

trainingData, testData = sdf.randomSplit([0.7, 0.3], seed=3456)

indexer = StringIndexer(inputCols=["product_category", "vine", "verified_purchase"], outputCols=["product_categoryIndex", "vineIndex",
"verified_purchaseIndex"], handleInvalid="keep")

encoder = OneHotEncoder(inputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex" ],
outputCols=["product_categoryVector", "vineVector", "verified_purchaseVector" ], dropLast=True, handleInvalid="keep")

assembler = VectorAssembler(inputCols=["product_categoryVector", "vineVector", "verified_purchaseVector", "total_votes",
"review_body_wordcount"], outputCol="features")

lr = LogisticRegression(maxIter=10)


reviews_pipe = Pipeline(stages=[indexer, encoder, assembler, lr])

grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0, 0.5, 1])

grid = grid.build()

print('Number of models to be tested: ', len(grid))

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

cv = CrossValidator(estimator=reviews_pipe,
estimatorParamMaps=grid,
evaluator=evaluator,
numFolds=3,
seed=789
)

cv = cv.fit(trainingData)

predictions = cv.transform(testData)

auc = evaluator.evaluate(predictions)
print('AUC:', auc)

predictions.groupby('label').pivot('prediction').count().fillna(0).show()
cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()
def calculate_precision_recall(cm):
tn = cm[0][1]
fp = cm[0][2]
fn = cm[1][1]
tp = cm[1][2]
precision = tp / ( tp + fp )
recall = tp / ( tp + fn )
accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
return accuracy, precision, recall, f1_score
print( calculate_precision_recall(cm) )

parammap = cv.bestModel.stages[3].extractParamMap()
for p, v in parammap.items():
print(p, v)

mymodel = cv.bestModel.stages[3]
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'r--')
x = mymodel.summary.roc.select('FPR').collect()
y = mymodel.summary.roc.select('TPR').collect()
plt.scatter(x, y)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.savefig("reviews_roc.png")


coeff = mymodel.coefficients.toArray().tolist()

var_index = dict()
for variable_type in ['numeric', 'binary']:
for variable in predictions.schema["features"].metadata["ml_attr"]["attrs"][variable_type]:
print("Found variable:", variable)
idx = variable['idx']
name = variable['name']
var_index[idx] = name # Add the name to the dictionary

for i in range(len(var_index)):
print(i, var_index[i], coeff[i])




