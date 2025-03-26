import tensorflow as tf
import random

# Chemin du fichier d'origine
tfrecords_filename = "./data/consumer-complaints.tfrecords"

# Définition des fichiers de sortie
train_filename = "./data/train.tfrecords"
val_filename = "./data/val.tfrecords"
test_filename = "./data/test.tfrecords"

# Lire le dataset TFRecord
dataset = list(tf.data.TFRecordDataset(tfrecords_filename))  # Convertir en liste pour faciliter le découpage
total_records = len(dataset)

# Définition des tailles
train_size = int(0.6 * total_records)
val_size = int(0.2 * total_records)
test_size = total_records - train_size - val_size  # Pour s'assurer de conserver le total

# Mélanger les données pour éviter le biais de séquence
random.shuffle(dataset)

# Répartir les données
train_data = dataset[:train_size]
val_data = dataset[train_size:train_size + val_size]
test_data = dataset[train_size + val_size:]

# Fonction pour écrire les données dans un fichier TFRecord
def write_tfrecord(filename, data):
    with tf.io.TFRecordWriter(filename) as writer:
        for record in data:
            writer.write(record.numpy())  # Convertir en bytes et écrire

# Écriture des fichiers
write_tfrecord(train_filename, train_data)
write_tfrecord(val_filename, val_data)
write_tfrecord(test_filename, test_data)

# Afficher les résultats
print(f"✅ Données divisées avec succès !")
print(f"Train: {train_size} enregistrements → {train_filename}")
print(f"Validation: {val_size} enregistrements → {val_filename}")
print(f"Test: {test_size} enregistrements → {test_filename}")
