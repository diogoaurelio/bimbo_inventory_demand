package com.groupo.bimbo

import java.io.File
import scala.io.Source

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}


case class Client(Cliente_ID:String, NombreCliente: String)
case class Product(Producto_ID:String, NombreProducto:String)
case class TestSchema(id: String, Semana:String, Agencia_ID:String, Canal_ID:String,
                      Ruta_SAK:String, Cliente_ID:String, Producto_ID:String)
case class TrainSchema(Semana:String, Agencia_ID:String, Canal_ID:String, Ruta_SAK:String, Cliente_ID:String,
                       Producto_ID:String, Venta_uni_hoy:String, Venta_hoy:String, Dev_uni_proxima:String,
                       Dev_proxima:String, Demanda_uni_equil:String)
case class Town(Agencia_ID:String, Town:String, State:String)


object bimbo  {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("krakenExplore")
    val sc = new SparkContext(conf) // sparkContext
    val ssc = new SQLContext(sc)  //sqlContext

    val s3 = "s3n://bonial-jobs-stage/notebooks/training/grupo_bimbo/"

    val s3Client = s"$s3/cliente_tabla.csv"
    lazy val clientRDD = sc.textFile(s3Client)
    val clientSplitter = clientRDD.map(line => line.split(",")).map { line =>
      Client(line(0).toString, line(1).toString)
    }
    val clientDf = ssc.createDataFrame(clientSplitter)

    val s3Product = s"$s3/producto_tabla.csv"
    lazy val productRDD = sc.textFile(s3Product)
    val productSplitter = productRDD.map(line => line.split(",")).map { line =>
      Product(line(0).toString, line(1).toString)
    }

    val s3Test = s"$s3/test.csv"
    lazy val testRDD = sc.textFile(s3Test)
    val testSplitter = testRDD.map(line => line.split(",")).map { line =>
      TestSchema(line(0).toString, line(1).toString, line(2).toString, line(3).toString,
        line(4).toString, line(5).toString, line(6).toString
      )
    }


    val s3Train = s"$s3/train.csv"


    val s3TownState = s"$s3/town_state.csv"






  }



}
