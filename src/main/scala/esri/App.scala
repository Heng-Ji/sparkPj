package esri

import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable
import scala.util.control.Breaks._

/**
  * Created By Heng Ji on 5/24/2017
  *
  */
object App {

    def main(args: Array[String]): Unit = {

        // initializing SparkContext and connect to hadoop directory
        val conf = new SparkConf()
          .setAppName("app")
          .setMaster("local[2]")
        val sc = new SparkContext(conf)

        // loading data
        val spark = SparkSession.builder()
          .config("spark.sql.warehouse.dir", "file:///C:/Users/heng9188/IdeaProjects/spark_project/spark-warehouse")
          .appName("app")
          .getOrCreate()


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

        import spark.implicits._

        // assemble longitude and latitude to features
        val assembler = new VectorAssembler()
          .setInputCols(Array("latitude", "longitude"))
          .setOutputCol("features")

        // transforming categorical data to numeric
        val indexer = new StringIndexer()
          .setInputCol("shape")
          .setOutputCol("shapeIndex")
          .fit(newData)
        val indexed = indexer.transform(newData)

        // geoData
        val geoData = assembler
          .transform(newData.select("latitude", "longitude"))
          .cache()

        println(geoData.count())

        // vector to array
        val toArray = udf[Array[Double], Vector](_.toArray)

        // kmeans clustering algorithm
        var model: KMeansModel = null
        val kmeans = new KMeans()

        val numClusters = 5
        val numIterations = 20
        kmeans.setK(5)
        model = kmeans.fit(geoData)
        //            data.map(datum => distToCentroid(datum)).mean()
        val a = model.clusterCenters.zipWithIndex.map {
            case (c, d) => (d, c.toArray)
        }
        val prediction = model.transform(geoData)
        val combined = indexed.as("d1")
          .join(prediction.as("d2"), col("d1.latitude") === col("d2.latitude"), "inner")
          .withColumn("label", col("prediction"))
          .drop(col("features"))

        val assemble = new VectorAssembler()
          .setInputCols(Array("duration", "shapeIndex"))
          .setOutputCol("features")

        val result = assemble
          .transform(combined)
          .select("features", "label")
          .cache()

        result.show()
        sys.exit()

//        val splits = result.randomSplit(Array(0.6, 0.4), seed = 1234L)
//        val train = splits(0)
//        val test = splits(1)
//        val layers = Array[Int](2, 3, 5, 5)
//
//        val trainer = new MultilayerPerceptronClassifier()
//          .setLayers(layers)
//          .setBlockSize(128)
//          .setSeed(1234L)
//          .setMaxIter(100)
//
//        //        train.show()
//        //        sys.exit()
//        val model1 = trainer.fit(train)
//        val results = model1.transform(test)
//        val predictionAndLabels = results.select("prediction", "label")
//        val evaluator = new MulticlassClassificationEvaluator()
//          .setMetricName("accuracy")
//
//        println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
//
//
//        sys.exit()

        //        val predictions = prediction.withColumn("features", toArray(prediction("features")))
        //        val clusterCentroid = sc.parallelize(a).toDF("predictions", "centroid")
        //        val combined = predictions.as("d1")
        //          .join(clusterCentroid.as("d2"), col("d1.prediction") === col("d2.predictions"), "inner")
        //          .select("predictions")


        //        combined.show()

//
//        // Euclidean distance
//        def distance(a: Vector, b: Vector): Double =
//            math.sqrt(a.toArray.zip(b.toArray).
//              map(p => p._1 - p._2).map(d => d * d).sum)
//
//        //         distance to centroid
//        //        def distToCentroid(datum: Row): Double = {
//        //            val cluster = model.predict(datum)
//        //            val centroid = model.clusterCenters(cluster)
//        //            distance(centroid, datum)
//        //        }
//
//
//        val calculation = udf[Double, Array[Double], Array[Double]] { (x, s) =>
//            math.sqrt(x.zip(s).map(a => a._1 - a._2).map(b => (b * b)).sum)
//        }
//
//
//        // score of datapoints to centroid, the less the more accurate
//        def clusteringScore(data: sql.DataFrame, k: Int): Double = {
//            val numClusters = k
//            val numIterations = 20
//            kmeans.setK(k)
//            model = kmeans.fit(data)
//            //            data.map(datum => distToCentroid(datum)).mean()
//            val a = model.clusterCenters.zipWithIndex.map {
//                case (c, d) => (d, c.toArray)
//            }
//            val prediction = model.transform(data)
//            val predictions = prediction.withColumn("features", toArray(prediction("features")))
//            val clusterCentroid = sc.parallelize(a).toDF("prediction", "centroid")
//            val combined = predictions.as("d1")
//              .join(clusterCentroid.as("d2"), col("d1.prediction") === col("d2.prediction"), "inner")
//
//            combined.printSchema()
//            combined.show()
//            val result1 = combined.select("features").rdd.map(row => row.get(0).asInstanceOf[mutable.WrappedArray[Double]].array).collect()
//            val result2 = combined.select("centroid").rdd.collect().map(row => row.get(0))
//
//            //            println(result2(0))
//            //            result1.filter(r => r.isInstanceOf[Array[Double]])
//                        result1.foreach(println)
////                        val foo: Nothing = result1
//            0
//        }
//
//                var k: Int = 0
//                val diff = 0.05
//                var prev: Double = 1000
//
////         searching for appropriate k
//                breakable {
//                    for (i <- 5 to 5) {
//                        val m = clusteringScore(geoData, i)
//                        if (prev > m && (prev - m) / prev < diff) {
//                            println((prev - m) / prev)
//                            k = i
//                            break
//                        }
//                        prev = m
//                    }
//                }
//        sys.exit()

        // patition geoData to each cluster
        //        val predictions = geoData.map { datum =>
        //            val prediction = model.predict(datum)
        //            (prediction, datum.toString)
        //        }.collect().sorted

        //        val patitioned: Array[A
    }

}
