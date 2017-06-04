package esri

import esri.App.{spark, sc}
import spark.implicits._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql
import org.apache.spark.sql.functions._

import scala.util.control.Breaks.{break, breakable}

/**
  * Created by heng9188 on 6/2/2017.
  */
object kMeans {
    // kmeans clustering algorithm
    var model: KMeansModel = null
    val kmeans = new KMeans()
    // score of datapoints to centroid, the less the more accurate


    // transform vector to array
    val toArray = udf[Array[String], Vector] {
        try {
            _.toArray.map(_.toString)
        } catch {
            case e: Exception => throw new Exception("\n" + "toArray failed")
        }
    }

    // transform String to double
    val toDouble = udf[Double, String] {
        try {
            _.toDouble
        } catch {
            case e: Exception => throw new Exception("\n" + "to Double failed")
        }
    }


    def toArrayDouble(x: Array[(Double, Int)], y: Int): Array[Double] = {
        try {
            x.filter(row => row._2 % 2 == y)
              .map(row => row._1)
        } catch {
            case e: Exception => throw new Exception("\n" + "tooArrayDouble failed")
        }
    }

    def dataframeToArray(x: sql.DataFrame, inputColumn: String, outputColumn: String): Array[Double] = {
        try {
            x.withColumn(outputColumn, toDouble(x(inputColumn)))
              .rdd.collect()
              .map(row => row.get(0)
                .toString
                .toDouble)
        } catch {
            case e: Exception => throw new Exception("\n" + "dataframeToArray failed")
        }
    }


    def clusteringScore(data: sql.DataFrame, k: Int): (Double, sql.DataFrame) = {

        kmeans.setK(k).setSeed(1L)
        model = kmeans.fit(data)

        // map cluster to numeric value, d is numeric value while c is the coordinates
        val clusterWithIndex = model.clusterCenters.zipWithIndex.map {
            case (c, d) => (d, c.toArray)
        }

        // predict numerical clusters, and transform clusterWithIndex to DataFrame (originally it was array)
        val prediction = model.transform(data)
        // change the type of features in prediction from vector to array for future usage
        val predictions = prediction
            .withColumn("features", toArray(prediction("features")))
        // transform clusterWithIndex to DataFrame, prediction contains the numerical value while centroid contains the coordinates
        val clusterCentroid = sc.parallelize(clusterWithIndex)
            .toDF("label", "centroid")

        // pairing the original coordinates with its corresponding centroid
        val combined = predictions.as("d1")
          .join(clusterCentroid.as("d2"), col("d1.prediction") === col("d2.label"), "inner")
          .drop(col("d1.prediction"))

        // extracting the coordinates
        val originalLatLon = combined.selectExpr("explode(features) as e")
        val clusterLatLon = combined.selectExpr("explode(centroid) as e")

        // paring lat and long
        val extractedOri = dataframeToArray(originalLatLon, "e", "features")
          .zipWithIndex


        val extractedClust = dataframeToArray(clusterLatLon, "e", "centroid")
          .zipWithIndex

        case class Row(i: Double, j: Double, k: Double, m: Double)
        val latCluster = toArrayDouble(extractedClust, 0)
        val longCluster = toArrayDouble(extractedClust, 1)
        val lat = toArrayDouble(extractedOri, 0)
        val long = toArrayDouble(extractedOri, 1)

        // transforming the four arrays to four DataFrame columns
        def zip4[A, B, C, D](l1: Array[A], l2: Array[B], l3: Array[C], l4: Array[D]) =
            l1.zip(l2).zip(l3).zip(l4).map { case (((a, b), c), d) => (a, b, c, d) }

        val xs = zip4(latCluster, longCluster, lat, long)
        val tmp = sc.parallelize(xs).toDF("latCluster", "longCluster", "lat", "long")

        // calculating the Euclidean distance
        val res = tmp.select($"latCluster" - $"lat" as "d1", $"longCluster" - $"long" as "d2")
        val square = res.select($"d1" * $"d1" as "d1", $"d2" * $"d2" as "d2")
        val sumation = square.select($"d1" + $"d2" as "d")
        sumation.show()
        val average = sumation.agg(sum("d").cast("long"))
        average.show()
        (math.sqrt(average.first().getLong(0)), combined.select("shape", "duration", "label"))
    }

    var k: Int = 0
    val diff = 0.03
    var prev: Double = Float.MaxValue.toDouble

    // searching for appropriate k
    def kMeansCluster(geoData: sql.DataFrame): (sql.DataFrame, Int) = {
        var result: sql.DataFrame = null
        breakable {
            for (i <- 5 to 5) {
                val (m, b) = clusteringScore(geoData, i)
                println(prev, m)
                if (prev > m && (prev - m) / prev < diff) {
                    k = i
                    result = b
//                    model.clusterCenters.foreach(println)
                    break
                }
                prev = m
            }
        }
        (result, k)
    }
}
