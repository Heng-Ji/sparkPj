package esri

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
import org.apache.spark.sql.types.{LongType, StructField, StructType}


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

    // add index of columns to dataframe
    def addColumnIndex(df: DataFrame) = spark.createDataFrame(
        df.rdd.zipWithIndex().map {
            case (row, idx) => Row.fromSeq(row.toSeq :+ idx)
        },
        StructType(df.schema.fields :+ StructField("columnIndex", LongType, false))
    )

    def main(args: Array[String]): Unit = {

        // initializing SparkContext and loading data


        import spark.implicits._

        val path = "complete1.csv"
        val df = spark.read
            .format("csv")
            .option("header", true)
            .csv(path)

        // data cleaning
        val data = df
            .selectExpr(
                "cast(datetime as string) datetime",
                "cast(shape as string) shape",
                "cast(duration as double) duration",
                "cast(latitude as double) latitude",
                "cast(longitude as double) longitude")

        // parse datetime to timestamp
        val ts = unix_timestamp($"datetime", "MM/dd/yyyy HH:mm").cast("timestamp")
        val dataToParse = data
            .withColumn("ts", ts)
            .na
            .drop()
            .filter("latitude != 0.0")
            .filter("duration != 0.0")
            .select("ts", "shape", "duration", "latitude", "longitude")
            .cache()

        val tsArr = dataToParse.select("ts").rdd.collect().map(row => row(0))

        val arr1 = (tsArr(0) +: tsArr.slice(0, tsArr.length - 1))
            .map(row => row.toString)

        val arr2 = tsArr.map(row => row.toString)
        val timeDiff = sc.parallelize(arr1.zip(arr2))
            .toDF("col1", "col2")
            .withColumn("col1", $"col1".cast("timestamp"))
            .withColumn("col2", $"col2".cast("timestamp"))
        val diffSecs = col("col2").cast("long") - col("col1").cast("long")


        val df2 = timeDiff.withColumn("diffMins", diffSecs / 60D)
        val df1WithIndex = addColumnIndex(dataToParse)
        val df2WithIndex = addColumnIndex(df2)
        val parsedData = df1WithIndex
            .join(df2WithIndex, Seq("columnIndex"))
            .drop("columnIndex", "ts", "col1", "col2")

        // assemble longitude and latitude to features
        val assembler = new VectorAssembler()
            .setInputCols(Array("latitude", "longitude"))
            .setOutputCol("features")

        // geoData
        val geoData = assembler
            .transform(parsedData)
            .cache()


        val (labeled, clusters) = kMeans.kMeansCluster(geoData)
//
//        labeled.show()
//        sys.exit()
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

        val vecToDouble = udf[Double, Vector] {
            try {
                _.toArray(0)
            } catch {
                case e: Exception => throw new Exception("\n" + "toDouble failed")
            }
        }


        val vecData = indexed
            .withColumn("duration", toVec(indexed("duration")))
            .withColumn("diffMins", toVec(indexed("diffMins")))

        val scalerModel = scaler.fit(vecData)
        val scaledData1 = scalerModel.transform(vecData)
        val transformedData1 = scaledData1
            .withColumn("duration", vecToDouble(scaledData1("durations")))

        scaler
            .setInputCol("diffMins")
            .setOutputCol("scaledDiff")
            .setWithStd(true)
            .setWithMean(false)
        val scalerModel2 = scaler.fit(transformedData1)
        val scaledData2 = scalerModel2.transform(transformedData1)
        val transformedData2 = scaledData2
            .withColumn("diffMins", vecToDouble(scaledData2("scaledDiff")))

        val assembler2 = new VectorAssembler()
            .setInputCols(Array("shapeIndex", "duration", "diffMins"))
            .setOutputCol("features")

        val trainingData = assembler2
            .transform(transformedData2)
            .select("features", "label")
            .withColumn("label", $"label".cast("double"))

        val splits = trainingData.randomSplit(Array(0.6, 0.4))
        val train = splits(0)
        val test = splits(1)
        val layers = Array[Int](3, 30, 30, clusters)

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
