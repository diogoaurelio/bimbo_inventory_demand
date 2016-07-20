
import AssemblyKeys._


lazy val root = (project in file("."))
  .settings(
    name := "'spark-bimbo",
    version := "1.0",
    scalaVersion := "2.11.8",
    mainClass in Compile := Some("com.berlinsmartdata")
  )

//val akkaVersion = "2.3.11"
val sparkVersion = "1.6.2"


libraryDependencies ++= Seq(
  //"com.typesafe.akka" %% "akka-actor"              % akkaVersion, // Akka Actor
  //"com.typesafe.akka" %% "akka-slf4j"              % akkaVersion, // Akka SLF4J
  "org.apache.spark"  %% "spark-core"              % sparkVersion,
  "org.apache.spark"  %% "spark-sql"               % sparkVersion, // Spark Dataframes
  "org.apache.spark"  %% "spark-mllib"             % sparkVersion // Spark MLLIB
  //"org.scalatest" % "scalatest_2.10" % "2.0" % "test",
  //"com.holdenkarau" % "spark-testing-base_2.10" % "1.5.0_1.4.0_1.4.1_0.1.2" % "test",
)

//resolvers += "scalaz-bintray" at "http://dl.bintray.com/scalaz/releases"

resolvers ++= Seq(
  "JBoss Repository" at "http://repository.jboss.org/nexus/content/repositories/releases/",
  "Spray Repository" at "http://repo.spray.cc/",
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Twitter4J Repository" at "http://twitter4j.org/maven2/",
  "Apache HBase" at "https://repository.apache.org/content/repositories/releases",
  "Twitter Maven Repo" at "http://maven.twttr.com/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "Mesosphere Public Repository" at "http://downloads.mesosphere.io/maven",
  Resolver.sonatypeRepo("public")
)

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)

assemblySettings

mergeStrategy in assembly := {
  case PathList("org", "apache", "spark", "unused", "UnusedStubClass.class") => MergeStrategy.first
  case x => (mergeStrategy in assembly).value(x)
}