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
import org.apache.spark.sql.expressions.Window

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
          .transform(newData)
          .cache()


        // kmeans clustering algorithm
        var model: KMeansModel = null
        val kmeans = new KMeans()

        //        val numClusters = 5
        //        val numIterations = 20
        //        kmeans.setK(5)
        //        model = kmeans.fit(geoData)
        //        //            data.map(datum => distToCentroid(datum)).mean()
        //        val a = model.clusterCenters.zipWithIndex.map {
        //            case (c, d) => (d, c.toArray)
        //        }
        //        val prediction = model.transform(geoData)
        //        val combined = indexed.as("d1")
        //          .join(prediction.as("d2"), col("d1.latitude") === col("d2.latitude"), "inner")
        //          .withColumn("label", col("prediction"))
        //          .drop(col("features"))
        //
        //        val assemble = new VectorAssembler()
        //          .setInputCols(Array("duration", "shapeIndex"))
        //          .setOutputCol("features")
        //
        //        val result = assemble
        //          .transform(combined)
        //          .select("features", "label")
        //          .cache()

        //        sys.exit()

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


        // score of datapoints to centroid, the less the more accurate
        def clusteringScore(data: sql.DataFrame, k: Int): (Double, sql.DataFrame) = {
            kmeans.setK(k)
            model = kmeans.fit(data)

            // map cluster to numeric value, d is numeric value while c is the coordinates
            val clusterWithIndex = model.clusterCenters.zipWithIndex.map {
                case (c, d) => (d, c.toArray)
            }

            // transform vector to array
            val toArray = udf[Array[String], Vector](_.toArray.map(_.toString))
            // transform String to double
            val toDouble = udf[Double, String](_.toDouble)

            // predict numerical clusters, and transform clusterWithIndex to DataFrame (originally it was array)
            val prediction = model.transform(data)
            // change the type of features in prediction from vector to array for future usage
            val predictions = prediction.withColumn("features", toArray(prediction("features")))
            // transform clusterWithIndex to DataFrame, prediction contains the numerical value while centroid contains the coordinates
            val clusterCentroid = sc.parallelize(clusterWithIndex).toDF("prediction", "centroid")

            // pairing the original coordinates with its corresponding centroid
            val combined = predictions.as("d1")
              .join(clusterCentroid.as("d2"), col("d1.prediction") === col("d2.prediction"), "inner")

            combined.show()
            // extracting the coordinates
            val originalLatLon = combined.selectExpr("explode(features) as e")
            val clusterLatLon = combined.selectExpr("explode(centroid) as e")

            // paring lat and long
            val extractedOri = originalLatLon
              .withColumn("centroid", toDouble(originalLatLon("e")))
              .rdd.collect()
              .map(row => row.get(0))
              .zipWithIndex
            val extractedClus = clusterLatLon
              .withColumn("features", toDouble(clusterLatLon("e")))
              .rdd.collect()
              .map(row => row.get(0))
              .zipWithIndex

            def seqToArray(x: Array[(_, Int)], y: Int): Array[Double] = {
                x.filter(row => row._2 % 2 == y)
                  .map(row => row._1.toString.toDouble)
            }

            case class Row(i: Double, j: Double, k: Double, m: Double)
            val latCluster = seqToArray(extractedClus, 0)
            val longCluster = seqToArray(extractedClus, 1)
            val lat = seqToArray(extractedOri, 0)
            val long = seqToArray(extractedOri, 1)

            // transforming the four arrays to four DataFrame columns
            def zip4[A, B, C, D](l1: Array[A], l2: Array[B], l3: Array[C], l4: Array[D]) =
                l1.zip(l2).zip(l3).zip(l4).map { case (((a, b), c), d) => (a, b, c, d) }

            val xs = zip4(latCluster, longCluster, lat, long)
            val tmp = sc.parallelize(xs).toDF("latCluster", "longCluster", "lat", "long")

            // calculating the Euclidean distance
            val res = tmp.select($"latCluster" - $"lat" as "d1", $"longCluster" - $"long" as "d2")
            val square = res.select($"d1" * $"d1" as "d1", $"d2" * $"d2" as "d2")
            val average = square.select(avg($"d1" + $"d2"))
            (average.head().getDouble(0), combined.select("shape", "duration", "prediction"))
        }

        var k: Int = 0
        val diff = 0.05
        var prev: Double = 1000

        // searching for appropriate k
        def kMeansCluster(geoData: sql.DataFrame): sql.DataFrame = {
            var n: sql.DataFrame = null
            breakable {
                for (i <- 5 to 5) {
                    val (m, n) = clusteringScore(geoData, i)
                    if (prev > m && (prev - m) / prev < diff) {
                        println((prev - m) / prev)
                        k = i
                        break
                    }
                    prev = m
                }
            }
            n
        }

    }

}
