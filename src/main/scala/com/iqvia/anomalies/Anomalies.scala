

package com.iqvia.anomalies

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object RunKMeans {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("data/germany-fact-data-labeled.csv").

      toDF(
        "wk_id",
        "trans_dt",
        "fcc_cd",
        "shop_cd",
        "trans_typ_nbr",
        "trans_qty",
        "trans_amt",
        "label")
//201701;20170105;814528;05368;11;4;66.12;1

    data.cache()

    val runKMeans = new RunKMeansAnomalies(spark)

  //  runKMeans.clustering1DefaultHyperParams(data)
  //  runKMeans.clustering2TuningHyperParams(data)
  //  runKMeans.clustering3Normalization(data) //normalization
  //  runKMeans.clustering4EntropyCalculus(data)
    runKMeans.buildAnomalyDetector(data)
    data.unpersist()
  }
}

class RunKMeansAnomalies(private val spark: SparkSession) {

  import spark.implicits._

  def clustering1DefaultHyperParams(data: DataFrame): Unit = {

    //counts the record by type
    data.select("label").groupBy("label").count().orderBy($"count".desc).show(10)

    //https://spark.apache.org/docs/2.4.0/ml-features.html#vectorassembler
    val inputCol = data.columns.filter(_ != "label")
    inputCol.foreach (println)

    //transformer (see below)
    val assembler = new VectorAssembler().
      setInputCols(inputCol).
      setOutputCol("featureVector")


  /* val outputDF = assembler.transform(data) //NOT NEEDED ONLY FOR VISUALIZATION
    outputDF.show(10, truncate = false)*/
  /*
  +------+--------+------+-------+-------------+---------+---------+-----+------------------------------------------------------+
  |wk_id |trans_dt|fcc_cd|shop_cd|trans_typ_nbr|trans_qty|trans_amt|label|featureVector                                         |
  +------+--------+------+-------+-------------+---------+---------+-----+------------------------------------------------------+
  |201701|20170103|453606|2330   |91           |4        |49.44    |1    |[201701.0,2.0170103E7,453606.0,2330.0,91.0,4.0,49.44] |
  */

   /* outputDF.select("featureVector").collect().take(10).foreach {
      row => println(row.toString())
    } */
    //featureVector
    //[[201701.0,2.0170103E7,453606.0,2330.0,91.0,4.0,49.44]]

    //https://spark.apache.org/docs/latest/ml-clustering.html#k-means
    //https://spark.apache.org/docs/latest/mllib-clustering.html
    //https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.KMeans
    //https://en.wikipedia.org/wiki/K-means_clustering
    /*
    setDefault(
    k -> 2,
    maxIter -> 20,
    initMode -> MLlibKMeans.K_MEANS_PARALLEL,
    initSteps -> 2,
    tol -> 1e-4,
    distanceMeasure -> DistanceMeasure.EUCLIDEAN)
     */

    //Estimator (see below)
    val kmeans = new KMeans().
    /*
    The Random Partition method first randomly assigns a cluster to each observation and then proceeds to the update step, thus computing the initial mean to be the centroid of the cluster's randomly assigned points. The Forgy method tends to spread the initial means out, while Random Partition places all of them close to the center of the data set

    Input Columns
    Param name	Type(s)	Default	Description
    featuresCol	Vector	"features"	Feature vector

    Output Columns
    Param name	Type(s)	Default	Description
    predictionCol	Int	"prediction"	Predicted cluster center
     */
      setSeed(Random.nextLong()).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    //https://spark.apache.org/docs/latest/ml-pipeline.html
    // pipeline are done with stages (transformers and estimators)
    /*
    DataFrame: This ML API uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types. E.g., a DataFrame could have different columns storing text, feature vectors, true labels, and predictions.

    Transformer: A Transformer is an algorithm which can transform one DataFrame into another DataFrame. E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.

    Estimator: An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

    Pipeline: A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow. using Stages

    Parameter: All Transformers and Estimators now share a common API for specifying parameters.
    */
    // assembler is a transformer (it contains a transform method)
    // kmeans is an estimator (it contains a fit method,  which accepts a DataFrame and produces a Model, which is a Transformer)
    // fit method to obtain kmeans model is not used directly but throw the pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
    //val pipelineModel = pipeline.fit(numericOnly)
    // pipeline fit internally is going to call to transform in assembler and to fit in kmeans
    val pipelineModel = pipeline.fit(data) //pipeline itself it is an estimator the will produce a model
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    // ^ both pipeline model and kmeans model are transformers

    println ("CLUSTER CENTERS:")
    kmeansModel.clusterCenters.foreach(println)

    //pipeline
    val withCluster = pipelineModel.transform(data) //withCluster is a dataframe

    println ("PREDICTIONS: ")
    withCluster.show (25, truncate = false)
    /*
    PREDICTIONS:
+------+--------+------+-------+-------------+---------+---------+-----+-------------------------------------------------------+-------+
|wk_id |trans_dt|fcc_cd|shop_cd|trans_typ_nbr|trans_qty|trans_amt|label|featureVector                                          |cluster|
+------+--------+------+-------+-------------+---------+---------+-----+-------------------------------------------------------+-------+
|201701|20170103|453606|2330   |91           |4        |49.44    |1    |[201701.0,2.0170103E7,453606.0,2330.0,91.0,4.0,49.44]  |1      |
|201701|20170105|11443 |5072   |32           |18       |135.9    |1    |[201701.0,2.0170105E7,11443.0,5072.0,32.0,18.0,135.9]  |1      |
|201701|20170104|313961|4169   |31           |2        |22.5     |1    |[201701.0,2.0170104E7,313961.0,4169.0,31.0,2.0,22.5]   |1      |
|201701|20170105|7631  |5984   |31           |2        |13.0     |1    |[201701.0,2.0170105E7,7631.0,5984.0,31.0,2.0,13.0]     |1      |
|201701|20170104|742041|11034  |31           |2        |16.38    |1    |[201701.0,2.0170104E7,742041.0,11034.0,31.0,2.0,16.38] |0      |
|201701|20170104|820041|2212   |31           |2        |16.9     |1    |[201701.0,2.0170104E7,820041.0,2212.0,31.0,2.0,16.9]   |0      |
     */

    println ("PREDICTION SUMMARY: ")
    withCluster.select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy($"cluster", $"count".desc).
      show(25)

    /*
    PREDICTION SUMMARY:
    +-------+-----+-----+
    |cluster|label|count|
    +-------+-----+-----+
    |      0|    1|60855|
    |      0|    0|    3|
    |      1|    1|39144|
    |      1|    0|    7|
    +-------+-----+-----+
     */
  }

  def clusteringScore1DefaultHyperParams(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]

    // A clustering could be considered good if each data point were near to its closest centroid
    // as more clusters are added points should be closer to the centroid. extreme case if k=num of records
    // every point will be their own cluster
    //
    // idea: 1. calculate the distance from a point to the centroid
    // 2. calculate the mean for all the distance from a point to a particular K
    // this provide a score of how good the cluster for a particular K is

    //Return the K-means cost (sum of squared distances of points to their nearest center) for this model on the given data.
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  def clusteringScore2TuningHyperParams(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")
    /*
    K-means is not necessarily able to find the optimal clustering for a given k.
    Its iterative process can converge from a random starting point to a local minimum,
    which may be good but not optimal. Accuracy Can be improved running the iteration longer and
    adjusting the tolerance (aka Epsilon, aka threshold) that controls the minimum amount of cluster centroid
    movement that is considered significant
    */

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector").
      setMaxIter(40). // increasing x2 the number of Iterations
      setTol(1.0e-5) // decreasing tolerance x10 (default) 1.0e-4

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  def clustering2TuningHyperParams(data: DataFrame): Unit = {

    // Rule of thumb to choose k ~ SQR (n/2) n=100009 => k = 224
    // warning features are not normalized!

    println ("DEFAULT VALUES FOR MaxIter= 20 and Tol= 1.0e-4")
    (20 to 300 by 20).map(k => (k, clusteringScore1DefaultHyperParams(data, k))).foreach(println)
    println ("VALUES FOR MaxIter= 40 and Tol= 1.0e-5")
    (20 to 300 by 20).map(k => (k, clusteringScore2TuningHyperParams(data, k))).foreach(println)

    /*
    DEFAULT VALUES FOR MaxIter= 20 and Tol= 1.0e-4
(20,2.0033689557731777E8)
(40,6.250672300988889E7)
(60,4.1578474254468545E7)
(80,3.342575625422024E7)
(100,2.720028351141748E7)
(120,2.481937148983936E7)
(140,2.1106864802836016E7)
(160,1.8217937951437734E7)
(180,1.7384442243862644E7)
(200,1.4963544938480964E7)
(220,1.3748166050119132E7)
(240,1.2487885181440387E7)
(260,1.1396593695661305E7)
(280,1.0797813533498537E7)
(300,1.0338333695597457E7)
VALUES FOR MaxIter= 40 and Tol= 1.0e-5
(20,1.9893361314398196E8)
(40,6.140755338349156E7)
(60,4.323854836238158E7)
(80,3.1869563695915356E7)
(100,3.2013278589104135E7)
(120,2.366689072741199E7)
(140,2.099021462175372E7)
(160,1.9395293301982407E7)
(180,1.6024866854517812E7)
(200,1.4750619039617797E7)
(220,1.359039473030718E7)
(240,1.2853203707111448E7)
(260,1.1763284925225936E7)
(280,1.0600367275125757E7)
(300,9715821.854588108)
     */
  }

  def clustering3Normalization(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    /* feature normalization: convert each feature to a standar score by subtraction the mean of the feature's
    value from each value and dividing by the standard desviation
     */

    /* Estimator
      StandardScaler transforms a dataset of Vector rows, normalizing each feature to have unit standard deviation and/or zero mean. It takes parameters:

      withStd: True by default. Scales the data to unit standard deviation.
      withMean: False by default. Centers the data with mean before scaling. It will build a dense output, so take care when applying to sparse input.
    */
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  def clustering3Normalization(data: DataFrame): Unit = {
    (100 to 400 by 20).map(k => (k, clustering3Normalization(data, k))).foreach(println)

    /*
(100,0.16770735082146013)
(120,0.14189028066526427)
(140,0.13160877896701986)
(160,0.11705448730963436)
(180,0.10659109526794978)
(200,0.09751485315005425)
(220,0.08890322208092712)
(240,0.0845856215275481)
(260,0.07925154946860839)
(280,0.07581899750693868)
(300,0.07183859995421085)
(320,0.06745382286718403)
(340,0.06509028530888612)
(360,0.062413392682350334)
(380,0.059366645227293825)
(400,0.05779961974009073)
     */
  }

  def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {

    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(assembler, scaler, kmeans))
    pipeline.fit(data)
  }

  def clusteringScore4EntropyCalculus(data: DataFrame, k: Int): Double = {
  // entropy in one cluster defined by count of the different labels that it has
    def entropy(counts: Iterable[Int]): Double = {

      println ("entropy counts  ============")
      println (counts.mkString(" | "))
      /*count of labels en each cluster (eg. 5 clusters)
        label 1   label 0
        7430    | 1   <- one each time
        12843
        53732   | 5
        1
        25993   | 4
        */

      val values = counts.filter(_ > 0)
      println ("entropy values  ============")
      println (values.mkString(" | "))

      val n = values.map(_.toDouble).sum

      println ("entropy n  ============")
      println (n)

      // map the label count in one cluster...
      val entropy = values.map { v =>
        val p = v / n
        -p * math.log(p)
      }.sum
      println ("entropy entropy  ============")
      println (entropy)
      entropy
      /*
        entropy counts  ============
        1
        entropy values  ============
        1
        entropy n  ============
        1.0
        entropy entropy  ============
        0.0

        entropy counts  ============
        99999 | 9
        entropy values  ============
        99999 | 9
        entropy n  ============
        100008.0
        entropy entropy  ============
        9.283419624445922E-4
       */

    }
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum

    println ("clusterlabel full DF==============")
    val clusterLabelFull = pipelineModel.transform(data) // dataframe
    clusterLabelFull.show(25, truncate=false)
    /*
    +------+--------+------+-------+-------------+---------+---------+-----+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+-------+
    |wk_id |trans_dt|fcc_cd|shop_cd|trans_typ_nbr|trans_qty|trans_amt|label|featureVector                                          |scaledFeatureVector                                                                                                       |cluster|
    +------+--------+------+-------+-------------+---------+---------+-----+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+-------+
    |201701|20170103|453606|2330   |91           |4        |49.44    |1    |[201701.0,2.0170103E7,453606.0,2330.0,91.0,4.0,49.44]  |[0.0,8667.669694129014,1.7930672073723162,0.6009112727014525,2.8003393448342098,0.19748238436239035,0.050201470429206925] |42     |
    |201701|20170105|11443 |5072   |32           |18       |135.9    |1    |[201701.0,2.0170105E7,11443.0,5072.0,32.0,18.0,135.9]  |[0.0,8667.670553586171,0.04523323777454755,1.3080781009192133,0.9847347146669749,0.8886707296307566,0.1379931195657205]   |9      |
    |201701|20170104|313961|4169   |31           |2        |22.5     |1    |[201701.0,2.0170104E7,313961.0,4169.0,31.0,2.0,22.5]   |[0.0,8667.670123857592,1.2410620086458728,1.0751927450181782,0.953961754833632,0.09874119218119518,0.022846542974457037]  |18     |
     */
    println ("clusterlabel======================")
    val clusterLabel = clusterLabelFull.select("cluster", "label").as[(Int, String)]
    clusterLabel.show (25,truncate = false) //DataSet

    /*
      +-------+-----+
      |cluster|label|
      +-------+-----+
      |42     |1    |
      |9      |1    |
      |18     |1    |
      |9      |1    |
      |21     |1    |
    */

    val weightedClusterEntropyGroup = clusterLabel.
    // Extract collections of labels, per cluster
    groupByKey { case (cluster, _) => cluster }

    //testing only!
    val z =weightedClusterEntropyGroup.mapGroups{case(k, iter) => (k, iter.toArray)}
    println ("weightedClusterEntropyGroup.mapGroups=======")
    z.show(truncate=false)
    //refer to #weightedClusterEntropyGroup.mapGroups======= in intermediate_results.txt

    weightedClusterEntropyGroup.count().show (truncate=false)
    /*
    +-----+--------+
    |value|count(1)|
    +-----+--------+
    |1    |26943   |
    |3    |23656   |
    |4    |33      |
    |2    |36038   |
    |0    |13339   |
    +-----+--------+
     */

   val weightedClusterEntropy = weightedClusterEntropyGroup.
      //maps each cluster
      mapGroups { case (_, clusterLabels) =>
      val labels = clusterLabels.map { case (_, label) => label }.toSeq
      //refer to #weightedClusterEntropyGroup.mapGroups: (at this point) in intermediate_result.txt

        // Count labels in collections
        val labelCounts = labels.groupBy(identity) //labels seq[String]
     /*   println ("labelCounts  ============")
        println (labelCounts.mkString(" | "))
        1 -> Stream(1, ?)
        1 -> Stream(1, ?) | 0 -> Stream(0, ?)
        1 -> Stream(1, ?) | 0 -> Stream(0, ?)
        0 -> Stream(0, ?)
        1 -> Stream(1, ?) | 0 -> Stream(0, ?)*/
        .values
        /*
        Stream(1, ?)
        Stream(1, ?) | Stream(0, ?)
        Stream(1, ?) | Stream(0, ?)
        Stream(1, ?) | Stream(0, ?)
        Stream(1, ?) | Stream(0, ?)
        */
        .map(_.size)
        /*count of labels en each cluster (eg. 5 clusters)
        label 1   label 0
        7430    | 1
        12843
        53732   | 5
        1
        25993   | 4
        */
      //println (labels.size)
      /*
      7431
      12843
      53737
      1
      25997
      */

     /*
     A good clustering should have clusters whose collections of labels are homogeneous, one metric
     for homogeneity is entropy, low entropy means that clustering is good. So a weighted average of entropy
     can be used as cluster score.

     Entropy concept cames from Information Theory. It captures how much uncertainty ("incertidumbre") the collection
     of target values in the subset contains. A subset containing one class only is completely certain,
     and has 0 entropy.

      see definition in def entropy
      */
        labels.size * entropy(labelCounts) //angel: weighting by cluster size
      }.collect()

    /*weightedClusterEntropy (for each cluster) --------------------------------------------------------------
    0.0 | 0.0 | 39.26637532612693 | 11.494560478396911 | 47.28944472820268
    */
    // Average entropy weighted by cluster size
  //angel: entropy for all the k clusters div by the number of points
    weightedClusterEntropy.sum / data.count()
  }

  def clustering4EntropyCalculus(data: DataFrame): Unit = {

    (100 to 400 by 20).map(k => (k, clusteringScore4EntropyCalculus(data, k))).foreach(println)
    /*
    (100,2.9603550193587426E-4)
    (120,2.8556669252159195E-4)
    (140,2.557831689753919E-4)
    (160,2.4205132161333814E-4)
    (180,2.599602872346715E-4)
    (200,2.505642016926496E-4)
    (220,2.601822787346768E-4)
    (240,2.2818569536202385E-4)
    (260,2.4373480115106713E-4)
    (280,1.8339071281034587E-4)
    (300,1.8793613197307901E-4)
    (320,1.5994217316693247E-4)
    (340,2.3410672108642986E-4)
    (360,2.1485261783729033E-4)
    (380,1.7098063079669802E-4)
    (400,1.5088534095466805E-4)
     */

    val pipelineModel = fitPipeline4(data, 220)
    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")
    countByClusterLabel.show(numRows = 250)
  }

  // Detect anomalies

  def buildAnomalyDetector(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 240)

    //320 5
    //280 5
    //260 4
    //250 4
    //248 4
    //245 6
    //240 6
    //238 4
    //235 5
    //230 5
    //224 5
    //2   6

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters
    //println ("centroids=======")
    //centroids.foreach(println)
    /*
    [0.0,8667.670087023736,3.4077701212281495,0.7175797463390932,0.9628293914223331,0.11179921106556447,0.024926202132020212]
    [0.0,8667.669694129014,2.1042843744812827,0.5109035327131234,2.831112304667553,246.85298045298794,10.154019099758683]
      ... (240)
     */

    val clustered = pipelineModel.transform(data)
   // println ("clustered=======")
   // clustered.show(truncate=false,numRows=5)
    /*
    +------+--------+------+-------+-------------+---------+---------+-----+------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+-------+
    |wk_id |trans_dt|fcc_cd|shop_cd|trans_typ_nbr|trans_qty|trans_amt|label|featureVector                                         |scaledFeatureVector                                                                                                       |cluster|
    +------+--------+------+-------+-------------+---------+---------+-----+------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+-------+
    |201701|20170103|453606|2330   |91           |4        |49.44    |1    |[201701.0,2.0170103E7,453606.0,2330.0,91.0,4.0,49.44] |[0.0,8667.669694129014,1.7930672073723162,0.6009112727014525,2.8003393448342098,0.19748238436239035,0.050201470429206925] |74     |
    |201701|20170105|11443 |5072   |32           |18       |135.9    |1    |[201701.0,2.0170105E7,11443.0,5072.0,32.0,18.0,135.9] |[0.0,8667.670553586171,0.04523323777454755,1.3080781009192133,0.9847347146669749,0.8886707296307566,0.1379931195657205]   |102    |
    |201701|20170104|313961|4169   |31           |2        |22.5     |1    |[201701.0,2.0170104E7,313961.0,4169.0,31.0,2.0,22.5]  |[0.0,8667.670123857592,1.2410620086458728,1.0751927450181782,0.953961754833632,0.09874119218119518,0.022846542974457037]  |181    |
    |201701|20170105|7631  |5984   |31           |2        |13.0     |1    |[201701.0,2.0170105E7,7631.0,5984.0,31.0,2.0,13.0]    |[0.0,8667.670553586171,0.030164715324440473,1.5432845733242453,0.953961754833632,0.09874119218119518,0.013200224829686288]|102    |
    |201701|20170104|742041|11034  |31           |2        |16.38    |1    |[201701.0,2.0170104E7,742041.0,11034.0,31.0,2.0,16.38]|[0.0,8667.670123857592,2.933227037618023,2.845688833900355,0.953961754833632,0.09874119218119518,0.016632283285404722]    |94     |
     */

    val thresholds = clustered.
      select("cluster", "scaledFeatureVector").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100)//.last

   // println ("thresholds size ======")
   // println (thresholds.size)

    //println ("thresholds=======")
    //thresholds.foreach(println)
    /*
    50.16104918267672
    45.400328707225555
    35.79931097577939
    22.250799617716584
      ... (100 in total)
    3.843270196841613
    3.766700575359129

     */

    val threshold = thresholds.last
    //println ("threshold =========")
    //println (threshold) 3.766700575359129

    val originalCols = data.columns
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail:_*)

    println("num anomalies======")
    println (anomalies.count) //100

    anomalies.show(100)
    /*
    +------+--------+------+-------+-------------+---------+---------+-----+
    | wk_id|trans_dt|fcc_cd|shop_cd|trans_typ_nbr|trans_qty|trans_amt|label|
    +------+--------+------+-------+-------------+---------+---------+-----+
    |201701|20170103|320722|   8336|           31|        2|  5000.98|    0|
    |201701|20170103|214268|   5204|           11|        2|  32000.9|    0|
    |201701|20170104|112008|   2710|           21|        2|  8000.26|    0|
    |201701|20161231|  8715|   7336|           31|        2|  20000.9|    0|
    |201701|20170104|230765|   4543|           21|        2| 24000.64|    0|
    |201701|20170104|829011|   2363|           91|        2| 27410.34|    1|
    |201701|20170103|532337|   2948|           31|        2|  40000.0|    0|
    |201701|20161231|793279|   4176|           11|        2|  2198.16|    1|
     */
  }
}