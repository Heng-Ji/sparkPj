package esri

import esri.App.{sc, spark}
import spark.implicits._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.functions._

import scala.util.control.Breaks.{break, breakable}

/**
  * Created by heng9188 on 6/2/2017.
  */
object kMeans {

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


    def clusteringScore(data: sql.DataFrame, k: Int): (Double, sql.DataFrame, sql.DataFrame, KMeansModel) = {

        val kmeans = new KMeans().setK(k).setMaxIter(40)
        //        data.show()
        val model = kmeans.fit(data)


        // map cluster to numeric value, d is numeric value while c is the coordinates
        val clusterWithIndex = model.clusterCenters.zipWithIndex.map {
            case (c, d) => (d, c.toArray)
        }
        val clusterCentroid = sc.parallelize(clusterWithIndex)
          .toDF("label", "centroid")

        // predict numerical clusters
        val prediction = model.transform(data)
        val result = prediction
          .withColumn("label", prediction("prediction"))
          .drop("features")

        // compute Euclidean distance
        val err = math.sqrt(model.computeCost(data) / data.count())
        println(err)
        (err, result, clusterCentroid.select("centroid"), model)
        //

        //        // pairing the original coordinates with its corresponding centroid
        //        val combined = predictions.as("d1")
        //          .join(clusterCentroid.as("d2"), col("d1.prediction") === col("d2.label"), "inner")
        //          .drop(col("d1.prediction"))
        //
        //        // extracting the coordinates
        //
        //        val originalLatLon = combined.selectExpr("explode(features) as e")
        //        val clusterLatLon = combined.selectExpr("explode(centroid) as e")
        //
        //        // paring lat and long
        //        val extractedOri = dataframeToArray(originalLatLon, "e", "features")
        //          .zipWithIndex
        //
        //        val extractedClust = dataframeToArray(clusterLatLon, "e", "centroid")
        //          .zipWithIndex
        //
        //        case class Row(i: Double, j: Double, k: Double, m: Double)
        //        val latCluster = toArrayDouble(extractedClust, 0)
        //        val longCluster = toArrayDouble(extractedClust, 1)
        //        val lat = toArrayDouble(extractedOri, 0)
        //        val long = toArrayDouble(extractedOri, 1)
        //
        //        // transforming the four arrays to four DataFrame columns
        //        def zip4[A, B, C, D](l1: Array[A], l2: Array[B], l3: Array[C], l4: Array[D]) =
        //            l1.zip(l2).zip(l3).zip(l4).map {
        //                case (((a, b), c), d) => (a, b, c, d)
        //            }
        //
        //        val xs = zip4(latCluster, longCluster, lat, long)
        //        val tmp = sc.parallelize(xs).toDF("latCluster", "longCluster", "lat", "long")
        //
        //        // calculating the Euclidean distance
        //        val res = tmp.select($"latCluster" - $"lat" as "d1", $"longCluster" - $"long" as "d2")
        //        val square = res.select($"d1" * $"d1" as "d1", $"d2" * $"d2" as "d2")
        //        val sumation = square.select($"d1" + $"d2" as "d")
        //
        //        val average = sumation.select(avg($"d"))
        //        println(model.computeCost(data)/data.count())
        //        sys.exit()

    }

    var k: Int = 0
    val diff = 0.03
    var prev: Double = Float.MaxValue.toDouble

    // searching for appropriate k 
    def kMeansCluster(geoData: sql.DataFrame): (sql.DataFrame, Int) = {
        var result: sql.DataFrame = null
        breakable {
            for (i <- 2 to 50) {
                val (err, r, c, model) = clusteringScore(geoData, i)
                if (prev > err && (prev - err) / prev < diff) {
                    k = i
                    result = r
                    val zipped = model.clusterCenters.flatMap(_.toArray).zipWithIndex
                    val lat = zipped.collect {
                        case (value, i) if i % 2 == 0 => value
                    }
                    val long = zipped.collect {
                        case (value, i) if i % 2 != 0 => value
                    }
                    val clusters = sc.parallelize(lat.zip(long)).toDF("latitude", "longitude")
                    clusters
//                    model.save("kmeans")
//                    result
                      .coalesce(1)
                      .write
                      .format("csv")
                      .option("header", "true")
                      .save("centroid.csv")
                    sys.exit()
                    break
                }
                prev = err
            }
        }
        (result, k)
    }
}
