package com.google.cloud.ml.samples.textclassification;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.cloud.bigquery.DatasetId;
import com.google.cloud.bigquery.DatasetInfo;
import com.google.cloud.bigquery.ExtractJobConfiguration;
import com.google.cloud.bigquery.Job;
import com.google.cloud.bigquery.JobInfo;
import com.google.cloud.bigquery.Table;
import com.google.cloud.bigquery.TableId;
import com.google.cloud.bigquery.TableInfo;
import com.google.cloud.bigquery.ViewDefinition;

import java.util.*;
import java.util.concurrent.TimeoutException;

import org.apache.avro.generic.GenericData;
import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.tensorflow.example.Example;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.Int64List;
import scala.Function1;
import scala.Tuple2;
import scala.collection.JavaConversions;
import scala.collection.Seq;

/**
 * Created by elibixby on 2/1/17.
 */
public class TextClassificationPreprocessor {

  static String QUERY = "SELECT subreddit, title FROM [fh-bigquery:reddit_posts.full_corpus_201512]" +
      "WHERE subreddit IN ( SELECT subreddit FROM (" +
      "SELECT subreddit, COUNT(subreddit) FROM [fh-bigquery:reddit_posts.full_corpus_201512]" +
      "GROUP BY subreddit ORDER BY 2 DESC LIMIT %i))";

  static String EXPORT_FILE_PATTERN = "/text_preprocessor/output*.avro";

  static String TEMP_DATASET = "temp_preprocessor";
  static String TEMP_TABLE = "temp_table";

  static class AssembleUDF implements UDF1<List<Map.Entry<Integer, Integer>>, List<Integer>>{

    public List<Integer> call(List<Map.Entry<Integer, Integer>> entries) throws Exception {
      ArrayList<Integer> wordIds = new ArrayList<Integer>(entries.size());
      for (Map.Entry<Integer, Integer> entry: entries){
        wordIds.add(entry.getKey(), entry.getValue());
      }
      return wordIds;      }
  }


  static void exportTable(String filePattern, int numClasses, String projectId)
      throws TimeoutException, InterruptedException {

    BigQuery bigquery = BigQueryOptions.getDefaultInstance().getService();
    DatasetId datasetId = DatasetId.of(projectId, TEMP_DATASET);
    com.google.cloud.bigquery.Dataset dataset = bigquery.getDataset(datasetId);

    if (dataset == null){
      dataset = bigquery.create(DatasetInfo.of(datasetId));
    }

    TableId tempTable = TableId.of(projectId, TEMP_DATASET, TEMP_TABLE);
    Table table = bigquery.create(TableInfo.of(
        tempTable,
        ViewDefinition.of(String.format(QUERY, numClasses))
    ));

    ExtractJobConfiguration jobConfig = ExtractJobConfiguration
        .newBuilder(tempTable, Collections.singletonList(filePattern))
        .setCompression("GZIP")
        .setFormat("AVRO")
        .build();

    Job extractJob = bigquery.create(JobInfo.of(jobConfig));

    extractJob.waitFor();
  }


  public static void main(String[] args) throws TimeoutException, InterruptedException {
    int numClasses = Integer.parseInt(args[1]);
    String basePath = args[2];
    SparkSession sp = SparkSession.builder().appName("TFTextClassifier").getOrCreate();
    SparkConf sc = sp.sparkContext().conf();
    String projectId = sc.get("fs.gs.project.id");
    String filePattern = sc.get("fs.default.name") + EXPORT_FILE_PATTERN;

    exportTable(filePattern, numClasses, projectId);
    
    Dataset<Row> df = sp.read().format("com.databricks.spark.avro").load(filePattern);

    Dataset<Row> tokenized = new Tokenizer()
            .setInputCol("title")
            .setOutputCol("words")
            .transform(df).na().drop();


    StringIndexerModel subredditIndex = new StringIndexer()
            .setInputCol("subreddit")
            .setOutputCol("subredditId")
            .fit(tokenized);
    sp.createDataFrame(Arrays.asList(subredditIndex.labels()), String.class)
            .write().csv(basePath + "subreddit_vocab.csv");

    subredditIndex.transform(tokenized);

    Dataset<Row> positions = tokenized.select(
            tokenized.col("id"),
            tokenized.col("subreddit"),
            functions.posexplode(tokenized.col("words")).as(new String[]{"pos", "word"})
    );

    StringIndexerModel wordIndex = new StringIndexer()
            .setInputCol("word")
            .setOutputCol("wordId")
            .fit(positions);
    sp.createDataFrame(Arrays.asList(subredditIndex.labels()), String.class)
            .write().csv(basePath + "vocab.csv");

    Dataset<Row> indexed = wordIndex.transform(positions);

    sp.sqlContext().udf().register(
            "assemble",
            new AssembleUDF(),
            DataTypes.createArrayType(DataTypes.IntegerType)
    );

    Dataset<Row> gathered = indexed.select(
            indexed.col("id"),
            indexed.col("subreddit"),
            functions.map(
                    indexed.col("pos"),
                    indexed.col("wordId")
            ).as("indexed")
    ).groupBy(indexed.col("id")).agg(
            functions.collect_list("indexed").as("indices")
    );

    Dataset<Row> squashed = gathered.select(
            gathered.col("subreddit"),
            functions.callUDF("assemble", gathered.col("indices")).as("wordIds")
    );


    squashed.map(new MapFunction<Row, byte[]>() {
      public byte[] call(Row row) throws Exception {
        Int64List.Builder words = Int64List.newBuilder().addAllValue(
                (List<Long>)(List<?>)row.getList(row.fieldIndex("wordIds"))
        );
        Int64List.Builder subreddit = Int64List.newBuilder().addValue(row.getLong(row.fieldIndex("subredditId")));
        Features.Builder features = Features.newBuilder()
                .putFeature("words", Feature.newBuilder().setInt64List(words).build())
                .putFeature("subreddit", Feature.newBuilder().setInt64List(subreddit).build());
        return Example.newBuilder().setFeatures(features).build().toByteArray();
      }
    }, Encoders.BINARY())
            .write()
            .format("org.tensorflow.hadoop.io.TFRecordFileOutputFormat")
            .save(basePath + "output_*.tfrecord.pb");
  }

}
