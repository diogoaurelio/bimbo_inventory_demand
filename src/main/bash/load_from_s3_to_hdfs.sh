#!/usr/bin/env bash

HDFS_ROOT="mnt"
HDFS_DIR="$HDFS_ROOT/tests/"
HOME_DIR="/$HDFS_DIR"

if [ -f "/$HOME_DIR/secrets.cfg" ]; then
    source /$HOME_DIR/secrets.cfg
else
    echo "Could NOT find /$HOME_DIR/secrets.cfg"
    exit 1
fi
hadoop fs -mkdir -p $HDFS_ROOT
hadoop fs -mkdir -p $HDFS_DIR

aws s3 cp $S3_BUCKET/train.csv $HOME_DIR/
hadoop fs -put $HOME_DIR/train.csv $HDFS_DIR/train.csv

aws s3 cp $S3_BUCKET/test.csv $HOME_DIR/
hadoop fs -put $HOME_DIR/test.csv $HDFS_DIR/test.csv

aws s3 cp $S3_BUCKET/cliente_tabla.csv $HOME_DIR/
hadoop fs -put $HOME_DIR/cliente_tabla.csv $HDFS_DIR/cliente_tabla.csv

aws s3 cp $S3_BUCKET/producto_tabla.csv $HOME_DIR/
hadoop fs -put $HOME_DIR/producto_tabla.csv $HDFS_DIR/producto_tabla.csv

aws s3 cp $S3_BUCKET/town_state.csv $HOME_DIR/
hadoop fs -put $HOME_DIR/town_state.csv $HDFS_DIR/town_state.csv


