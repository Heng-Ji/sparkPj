package esri

import esri.App.{indexer, scaler}
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.dmg.pmml.TimeSeries

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
    spark.sparkContext.setLogLevel("ERROR")

    // add index of rows to dataframe
    def addRowIndex(df: DataFrame) = spark.createDataFrame(
        df.rdd.zipWithIndex().map {
            case (row, idx) => Row.fromSeq(row.toSeq :+ idx)
        },
        StructType(df.schema.fields :+ StructField("columnIndex", LongType, false))
    )

    // assemble the columns to vector for model to fit
    def assembler(df: sql.DataFrame, inputCol: Array[String]): sql.DataFrame = {
        val assembler = new VectorAssembler()
          .setInputCols(inputCol)
          .setOutputCol("features")
        assembler.transform(df)
    }

    // normalizing duration to unit mean, zero deviation
    def scaler(df: sql.DataFrame, inputCol: String, outputCol: String): sql.DataFrame = {
        val scaler = new StandardScaler()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
          .setWithStd(true)
          .setWithMean(false)
        val scalerModel = scaler.fit(df)
        scalerModel.transform(df)
    }

    def indexer(df: sql.DataFrame, inputCol: String, outputCol: String): sql.DataFrame = {
        val indexer = new StringIndexer()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
          .fit(df)
        indexer.transform(df)
    }

    // transform DataFrame column with type double to type Vec
    val doubleToVec = udf[Vector, Double] {
        try {
            Vectors.dense(_)
        } catch {
            case e: Exception => throw new Exception("\n" + "toVec failed")
        }
    }

    //    // transform Vec back to double
    //    val vecToDouble = udf[Double, Vector] {
    //        try {
    //            _.toArray(0)
    //        } catch {
    //            case e: Exception => throw new Exception("\n" + "toDouble failed")
    //        }
    //    }

    def main(args: Array[String]): Unit = {

        // initializing SparkContext and loading data
        import spark.implicits._
//        lr.lr()
//        sys.exit()
        val path = "complete1.csv"
        val df = spark.read
          .format("csv")
          .option("header", true)
          .csv(path)

        // data cleaning
        val data = df
          .selectExpr("cast (state as string) state",
              "cast(datetime as string) datetime",
              "cast(shape as string) shape",
              "cast(duration as double) duration",
              "cast(latitude as double) latitude",
              "cast(longitude as double) longitude")

        val tmp = data.na.drop().filter("latitude != 0.0").filter("duration != 0.0")
        // parse datetime to timestamp
        val ts = unix_timestamp($"datetime", "MM/dd/yyyy HH:mm").cast("timestamp")
        val dataToParse = data
          .withColumn("ts", ts)
          .na
          .drop()
          .filter("latitude != 0.0")
          .filter("duration != 0.0")
          .select("ts", "state", "shape", "duration", "latitude", "longitude")
          .orderBy(asc("ts"))
        dataToParse.cache()
//        sys.exit()


        // transform array of timestamp from any(type) to string
        val tsArr = dataToParse.select("ts").rdd.collect().map(row => row(0))

        // split timestamp array to two arrays, one indexed from [0,0,1,2,3,...,n-2],
        // another from [0,1,2,3,...,n-1], then create dataframe consisting of 2 columns,
        // col1 and col2, where col1 stands for arr1 and col2 stands for arr2, apply
        // col2 - col1 to calculate the time difference
        val arr1 = (tsArr(0) +: tsArr.slice(0, tsArr.length - 1))
          .map(row => row.toString)
        val arr2 = tsArr.map(row => row.toString)

        val timeDiff = sc.parallelize(arr1.zip(arr2))
          .toDF("col1", "col2")
          .withColumn("col1", $"col1".cast("timestamp"))
          .withColumn("col2", $"col2".cast("timestamp"))
        // function to calculate time difference in seconds
        val diffSecs = col("col2").cast("long") - col("col1").cast("long")
        // df2 with column "diffMins", containing the time difference between two consecutive datetime
        val df2 = timeDiff.withColumn("diffMins", diffSecs / 60D)
        // add row index to df2 and original data, for the purpose of combining them (parsedData)
        val df1WithIndex = addRowIndex(dataToParse)
        val df2WithIndex = addRowIndex(df2)
        val parsedData = df1WithIndex
          .join(df2WithIndex, Seq("columnIndex"))
          .drop("columnIndex", "ts", "col1", "col2")


        // assemble longitude and latitude to features, in order to fit kMeansModel
        val geoData = assembler(parsedData, Array("longitude", "latitude"))
        geoData.cache()
                        kMeans.kMeansCluster(geoData)

        // loading kMeansModel and data
        val kmeansModel = KMeansModel.load("data/kmeans")
        val clusters = kmeansModel.clusterCenters.length
        val labeled = spark.read
          .format("csv")
          .option("header", true)
          .csv("data/res.csv")
          .drop("prediction")
          .withColumn("duration", $"duration".cast("double"))
          .withColumn("latitude", $"latitude".cast("double"))
          .withColumn("longitude", $"longitude".cast("double"))
          .withColumn("diffMins", $"diffMins".cast("double"))
          .withColumn("label", $"label".cast("int"))
        labeled.show()
        labeled.printSchema()
//          .withColumn("label", $"label".cast("int"))
//        for (i <- 0 to clusters - 1) {
//            labeled.filter($"label" === i)
//              .select("latitude", "longitude")
//              .coalesce(1, true)
//              .write
//              .format("csv")
//              .option("header", "true")
//              .save(s"res$i.csv")
//        }
//        sys.exit()

        // transforming categorical data to numeric

        val indexed = indexer(labeled, "shape", "shapeIndex")

        // transform duration and diffMins to vector in order to fit the scaler
        val vecData = indexed
          .withColumn("duration", doubleToVec(indexed("duration")))
          .withColumn("diffMins", doubleToVec(indexed("diffMins")))
//        val scaledDuration = scaler(vecData, "duration", "durations")
//
//        val scaledFinal = scaler(scaledDuration, "diffMins", "scaledDiff")

        val trainingData = assembler(vecData, Array("shapeIndex", "durations", "scaledDiff"))
          .select("features", "label")
        trainingData.show()
        trainingData.printSchema()
        sys.exit()

        // training neural networks
        val splits = trainingData.randomSplit(Array(0.6, 0.4))
        val train = splits(0)
        val test = splits(1)
        val layers = Array[Int](3, (3 + clusters) / 2, clusters)

        val trainer = new MultilayerPerceptronClassifier()
          .setLayers(layers)
          .setBlockSize(128)
          .setMaxIter(100)
          .setTol(0.1)

        //        val foo: Nothing = trainer

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
        val results = model1.bestModel.transform(test)
        val predictionAndLabels = results.select("prediction", "label")
        results
          .select("prediction", "label")
          .coalesce(1)
          .write
          .format("csv")
          .option("header", "true")
          .save("data/resForNN.csv")

        val inputRaw = spark.createDataFrame(Seq(
            (0.0, 1.1, 1.2),
            (1.0, 1.2, 0.5),
            (1.0, 0.8, 0.4)))
          .toDF("col1", "col2", "col3")
        val inputTransformed = assembler(inputRaw, Array("col1", "col2", "col3"))
        val newPrediction = model1.transform(inputTransformed)
        newPrediction.show()

        println("Test set accuracy for NN = " + evaluator.evaluate(predictionAndLabels))


    }

}
