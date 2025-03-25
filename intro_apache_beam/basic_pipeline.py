import re

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


input_file = "./intro_apache_beam/kinglear.txt" # 1/ The text is stored in intro_apache_beam/ folder
output_file = "./intro_apache_beam/output.txt"

# Define pipeline options object.
pipeline_options = PipelineOptions()


with beam.Pipeline(options=pipeline_options) as p: # 2/ Set up the Apache Beam pipeline.

    # Read the text file[pattern] into a PCollection.
    lines = p | ReadFromText(input_file) # 3/ Create a data collection by reading the text file

    # Count the occurrences of each word.
    counts = ( # 4/ Perform the transformations on the collection.
        lines
        | 'Split' >> (beam.FlatMap(lambda x: re.findall(r'[A-Za-z\']+', x)))
                      # .with_output_types(unicode))
        | 'PairWithOne' >> beam.Map(lambda x: (x, 1))
        | 'GroupAndSum' >> beam.CombinePerKey(sum))

    # Format the counts into a PCollection of strings.
    def format_result(word_count):
        (word, count) = word_count
        return "{}: {}".format(word, count)

    output = counts | 'Format' >> beam.Map(format_result)

    # Write the output using a "Write" transform that has side effects.
    output | WriteToText(output_file)