import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Charger le dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Sauvegarder temporairement le dataset en CSV
input_file = "./intro_apache_beam/iris_data.csv"
output_file = "./intro_apache_beam/iris_prepared.csv"
df.to_csv(input_file, index=False)

# Fonction de transformation : normaliser les features
class NormalizeFeatures(beam.DoFn):
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()

    def setup(self):
        # Lire le dataset initial pour ajuster le scaler
        df = pd.read_csv(input_file)
        self.scaler.fit(df[self.feature_columns])

    def process(self, row):
        df_row = pd.DataFrame([row], columns=self.feature_columns + ['target'])
        df_row[self.feature_columns] = self.scaler.transform(df_row[self.feature_columns])
        yield df_row.to_dict(orient="records")[0]

# Options du pipeline
pipeline_options = PipelineOptions()

# Définition du pipeline Apache Beam
with beam.Pipeline(options=pipeline_options) as p:
    (
        p
        | "Read CSV" >> beam.io.ReadFromText(input_file, skip_header_lines=1)
        | "Parse CSV" >> beam.Map(lambda line: dict(zip(df.columns, line.split(","))))
        | "Normalize Features" >> beam.ParDo(NormalizeFeatures(feature_columns=iris.feature_names))
        | "Format Output" >> beam.Map(lambda row: ",".join(map(str, row.values())))
        | "Write to CSV" >> beam.io.WriteToText(output_file, file_name_suffix=".csv")
    )

print(f"✅ Transformation terminée ! Résultat dans {output_file}")
