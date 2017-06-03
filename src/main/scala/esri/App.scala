package esri

import esri.kMeans
import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created By Heng Ji on 5/24/2017
  *
  */
object App {

    val conf = new SparkConf()
      .setAppName("app")
      .setMaster("local[2]")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder()
      .config("spark.sql.warehouse.dir", "file:///C:/Users/heng9188/IdeaProjects/spark_project/spark-warehouse")
      .appName("app")
      .getOrCreate()

    def main(args: Array[String]): Unit = {

        // initializing SparkContext and loading data


        import spark.implicits._

        val path = "../complete1.csv"
        val df = spark.read
          .format("csv")
          .option("header", true)
          .csv(path)

        // data cleaning
        val newData = df
          .selectExpr("cast(shape as string) shape",
              "cast(duration as double) duration",
              "cast(latitude as double) latitude",
              "cast(longitude as double) longitude")
          .na
          .drop()
          .filter("latitude != 0.0")
          .filter("duration != 0.0")

        // assemble longitude and latitude to features
        val assembler = new VectorAssembler()
          .setInputCols(Array("latitude", "longitude"))
          .setOutputCol("features")




        // geoData
        val geoData = assembler
          .transform(newData)
          .cache()

        val (labeled, clusters) = kMeans.kMeansCluster(geoData)
        // transforming categorical data to numeric
        val indexer = new StringIndexer()
          .setInputCol("shape")
          .setOutputCol("shapeIndex")
          .fit(labeled)

        val indexed = indexer.transform(labeled)

        // normalizing duration to unit mean, zero deviation
        val scaler = new StandardScaler()
          .setInputCol("duration")
          .setOutputCol("durations")
          .setWithStd(true)
          .setWithMean(false)

        val toVec = udf[Vector, Double] {
            try {
                Vectors.dense(_)
            } catch {
                case e: Exception => throw new Exception("\n" + "toVec failed")
            }
        }

        val toDouble = udf[Double, Vector] {
            try {
                _.toArray(0)
            } catch {
                case e: Exception => throw new Exception("\n" + "toDouble failed")
            }
        }

        val vecData = indexed
          .withColumn("duration", toVec(indexed("duration")))

        val scalerModel = scaler.fit(vecData)
        val scaledData = scalerModel.transform(vecData)
        val transformedData = scaledData
          .withColumn("duration", toDouble(scaledData("durations")))

        assembler
          .setInputCols(Array("shapeIndex", "duration"))
          .setOutputCol("features")

        val trainingData = assembler
          .transform(transformedData)
          .select("features", "label")

        val splits = trainingData.randomSplit(Array(0.6, 0.4))
        val train = splits(0)
        val test = splits(1)
        val layers = Array[Int](2, 30, 20, clusters)

        val trainer = new MultilayerPerceptronClassifier()
          .setLayers(layers)
          .setBlockSize(128)
          .setSeed(1234L)
          .setMaxIter(100)

        val pipeline = new Pipeline()
          .setStages(Array(trainer))

        //        train.show()
        //        sys.exit()

        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")

        val paramGrid = new ParamGridBuilder().build()

        val cv = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(5)

        val model1 = cv.fit(train)
        val results = model1.transform(test)
        val predictionAndLabels = results.select("prediction", "label")
        results.show()

        println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
        println("Number of clusters = " + clusters)

    }

}
