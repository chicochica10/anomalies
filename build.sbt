name := "anomalies"

version := "0.1"

scalaVersion := "2.12.8"


libraryDependencies ++= Seq(
  //add provided for spark dependencies when uploading
  "org.apache.spark" % "spark-core_2.12" % "2.4.0",// % "provided",
  "org.apache.spark" % "spark-sql_2.12" % "2.4.0",// % "provided",
  "org.apache.spark" % "spark-mllib_2.12" % "2.4.0"//, % "provided"
  //"org.apache.spark" % "spark-hive_2.12" % "2.4.0" % "provided"
  //,
  //"com.databricks" % "spark-csv_2.10" % "1.4.0"

)
