# %%
import tensorflow_data_validation as tfdv

# %%
# The first step in our data validation process is to generate some summary statistics for our data.
stats_from_csv = tfdv.generate_statistics_from_csv(
    data_location='./../data/consumer_complaints_with_narrative.csv',
    delimiter=',')

print(stats_from_csv)

# %%
# We can generate feature statistics from TFRecord files as well.
stats_from_tfrecord = tfdv.generate_statistics_from_tfrecord(
    data_location='./../data/consumer-complaints.tfrecords')

print(stats_from_tfrecord)

# %%
# the next step is to generate a schema of our dataset.
schema = tfdv.infer_schema(stats_from_csv)
tfdv.display_schema(schema)

# %%
# Comparing training and validation datasets in order to determine how representative the validation set is in regards to the training set. 
# Does the validation data follow our training data schema?
train_tfrecord_filename = './../data/train.tfrecords'
val_tfrecord_filename = './../data/val.tfrecords'

train_stats = tfdv.generate_statistics_from_tfrecord(
    data_location=train_tfrecord_filename
)
val_stats = tfdv.generate_statistics_from_tfrecord(
    data_location=val_tfrecord_filename
)
tfdv.visualize_statistics(
    lhs_statistics=val_stats, 
    rhs_statistics=train_stats,
    lhs_name='VAL_DATASET', 
    rhs_name='TRAIN_DATASET'
)
# %%
# We can detect anomalies in our validation dataset by comparing it to the schema of our training dataset.
train_stats = tfdv.generate_statistics_from_tfrecord(data_location=train_tfrecord_filename)
schema = tfdv.infer_schema(train_stats)
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=schema)
tfdv.display_anomalies(anomalies)

# %%
# Compare the skew between datasets
tfdv.get_feature(
    schema, 
    'company'
).skew_comparator.infinity_norm.threshold = 0.01
val_stats = tfdv.generate_statistics_from_tfrecord(
    data_location=val_tfrecord_filename
)
skew_anomalies = tfdv.validate_statistics(
    statistics=train_stats, 
    schema=schema, 
    serving_statistics=val_stats
)
tfdv.display_anomalies(skew_anomalies)
# %%
# Drift Anomaly Detection
# TFDV also provides a drift_comparator for comparing the statistics of 
# two datasets of the same type, such as two training sets 
# collected on two different days.

# tfdv.get_feature(schema,
#                  'company').drift_comparator.infinity_norm.threshold = 0.01
# drift_anomalies = tfdv.validate_statistics(statistics=train_stats_today,
#                                            schema=schema,
#                                            previous_statistics=\
#                                                train_stats_yesterday)

tfdv.get_feature(
    schema, 'company'
).drift_comparator.infinity_norm.threshold = 0.01

drift_anomalies = tfdv.validate_statistics(
    statistics=train_stats, 
    schema=schema, 
    previous_statistics=val_stats
)

tfdv.display_anomalies(drift_anomalies)
# %%
