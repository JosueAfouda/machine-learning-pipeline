""" Example module to convert csv data to TFRecords
"""

############## 1/ 1️⃣ Importations des librairies nécessaires ##############

import csv # Permet de lire le fichier CSV.

import tensorflow as tf # Nécessaire pour manipuler les TFRecords.
from tqdm import tqdm #  Ajoute une barre de progression pour suivre le traitement.


################# 2/ Fonctions pour convertir les valeurs en format TFRecords #################

# Convertit une chaîne de caractères (string) en un TFRecords Feature au format bytes_list.
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

# Convertit un entier (int) en un TFRecords Feature au format int64_list.
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

################# 3/ Nettoyage des données #################

# Remplace les valeurs manquantes dans la colonne zip_code par "99999".
def clean_rows(row):
    if not row["zip_code"]:
        row["zip_code"] = "99999"
    return row

# Remplace "XX" par "00" dans les codes postaux avant de les convertir en entiers.
def convert_zipcode_to_int(zipcode):
    if isinstance(zipcode, str) and "XX" in zipcode:
        zipcode = zipcode.replace("XX", "00")
    int_zipcode = int(zipcode)
    return int_zipcode

################# 4/ Définition des fichiers d'entrée et de sortie #################

original_data_file = "./data/consumer_complaints_with_narrative.csv" # Emplacement du fichier CSV à convertir.
tfrecords_filename = "consumer-complaints.tfrecords" # Nom du fichier de sortie TFRecords.
tf_record_writer = tf.io.TFRecordWriter(tfrecords_filename) # Ouvre un fichier TFRecords en mode écriture.

################# 5/ Lecture du CSV et écriture des TFRecords #################


with open(original_data_file) as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
    for row in tqdm(reader):
        row = clean_rows(row)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "product": _bytes_feature(row["product"]),
                    "sub_product": _bytes_feature(row["sub_product"]),
                    "issue": _bytes_feature(row["issue"]),
                    "sub_issue": _bytes_feature(row["sub_issue"]),
                    "state": _bytes_feature(row["state"]),
                    "zip_code": _int64_feature(convert_zipcode_to_int(row["zip_code"])),
                    "company": _bytes_feature(row["company"]),
                    "company_response": _bytes_feature(row["company_response"]),
                    "timely_response": _bytes_feature(row["timely_response"]),
                    "consumer_disputed": _bytes_feature(row["consumer_disputed"]),
                }
            )
        )
        tf_record_writer.write(example.SerializeToString())
    tf_record_writer.close()

