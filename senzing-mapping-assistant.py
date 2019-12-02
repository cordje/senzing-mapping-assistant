#! /usr/bin/env python

# -----------------------------------------------------------------------------
# senzing-mapping-assisant.py
# -----------------------------------------------------------------------------

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import pickle

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

__all__ = []
__version__ = 1.0
__date__ = '2018-10-29'
__updated__ = '2019-05-19'

SENZING_PRODUCT_ID = "9999"  # See https://github.com/Senzing/knowledge-base/blob/master/lists/senzing-product-ids.md
log_format = '%(asctime)s %(message)s'

# The "configuration_locator" describes where configuration variables are in:
# 1) Command line options, 2) Environment variables, 3) Configuration files, 4) Default values

configuration_locator = {
    "input_directory": {
        "default": "senzing-mapping-assistant-prepare",
        "env": "SENZING_INPUT_DIRECTORY",
        "cli": "input-directory"
    },
    "input_file": {
        "default": None,
        "env": "SENZING_INPUT_FILE",
        "cli": "input-file"
    },
    "jsonlines_file": {
        "default": None,
        "env": "SENZING_JSONLINES_FILE",
        "cli": "jsonlines-file"
    },
    "model_file": {
        "default": "model.pickle",
        "env": "SENZING_MODEL_FILE",
        "cli": "model-file"
    },
    "output_directory": {
        "default": "senzing-mapping-assistant-prepare",
        "env": "SENZING_OUTPUT_DIRECTORY",
        "cli": "output-directory"
    },
    "pretty": {
        "default": False,
        "env": "SENZING_PRETTY",
        "cli": "pretty"
    },
    "test_phrase": {
        "default": None,
        "env": "SENZING_TEST_PHRASE",
        "cli": "test-phrase"
    }
}

senzing_lexicon = {
    'account_domain': {},
    'account_number': {},
    'addr_city': {},
    'addr_country': {},
    'addr_from_date': {},
    'addr_full': {},
    'addr_line1': {},
    'addr_line2': {},
    'addr_line3': {},
    'addr_line4': {},
    'addr_line5': {},
    'addr_line6': {},
    'addr_postal_code': {},
    'addr_state': {},
    'addr_thru_date': {},
    'addr_type': {},
    'citizenship': {},
    'data_source': {},
    'date_of_birth': {},
    'drivers_license_number': {},
    'drivers_license_state': {},
    'email_address': {},
    'gender': {},
    'name_first': {},
    'name_full': {},
    'name_last': {},
    'name_middle': {},
    'name_org': {},
    'name_prefix': {},
    'name_suffix': {},
    'name_type': {},
    'nationality': {},
    'nin_country': {},
    'nin_number': {},
    'other_id_country': {},
    'other_id_number': {},
    'other_id_type': {},
    'passport_country': {},
    'passport_number': {},
    'phone_from_date': {},
    'phone_number': {},
    'phone_thru_date': {},
    'phone_type': {},
    'record_id': {},
    'related_from_date': {},
    'related_thru_date': {},
    'relationship_key': {},
    'relationship_role': {},
    'relationship_type': {},
    'social_handle': {},
    'social_network': {},
    'ssn_last4': {},
    'ssn_number': {},
    'tax_id_country': {},
    'tax_id_number': {},
    'tax_id_type': {},
    'website_address': {},
}

# -----------------------------------------------------------------------------
# Define argument parser
# -----------------------------------------------------------------------------


def get_parser():
    '''Parse commandline arguments.'''

    parser = argparse.ArgumentParser(prog="senzing-mapping-assistant.py", description="Spike for K-nearest neighbor")
    subparsers = parser.add_subparsers(dest='subcommand', help='Subcommands (SENZING_SUBCOMMAND):')

    subparser_1 = subparsers.add_parser('prepare', help='Read JSONLines file and output one file per JSON key.')
    subparser_1.add_argument("--jsonlines-file", dest="jsonlines_file", metavar="SENZING_JSONLINES_FILE", help="JSONLINES files. No default.")
    subparser_1.add_argument("--output-directory", dest="output_directory", metavar="SENZING_OUTPUT_DIRECTORY", help="Directory for output files. Default:senzing-mapping-assistant-prepare")

    subparser_2 = subparsers.add_parser('train', help='Create a model from training.')
    subparser_2.add_argument("--input-directory", dest="input_directory", metavar="SENZING_INPUT_DIRECTORY", help="Output directory from prepare step. Default: senzing-mapping-assistant-prepare")
    subparser_2.add_argument("--model-file", dest="model_file", metavar="SENZING_MODEL_FILE", help="Output filename of model created by training. Default: model.pickle")

    subparser_3 = subparsers.add_parser('test-phrase', help='Test a phrase.')
    subparser_3.add_argument("--test-phrase", dest="test_phrase", metavar="SENZING_TEST_PHRASE", help="Phrase to test. No default.")
    subparser_3.add_argument("--model-file", dest="model_file", metavar="SENZING_MODEL_FILE", help="Output filename of model created by training. Default: model.pickle")

    subparser_4 = subparsers.add_parser('suggest', help='Suggest a category.')
    subparser_4.add_argument("--input-file", dest="input_file", metavar="SENZING_INPUT_FILE", help="File to analyze. No default.")
    subparser_4.add_argument("--model-file", dest="model_file", metavar="SENZING_MODEL_FILE", help="Filename of the trained model. Default: model.pickle")
    subparser_4.add_argument("--pretty", dest="pretty", action="store_true", help="Pretty print output. Default: False (SENZING_PRETTY)")

    subparser_5 = subparsers.add_parser('suggest-as-markdown', help='Suggest a category.')
    subparser_5.add_argument("--input-file", dest="input_file", metavar="SENZING_INPUT_FILE", help="File to analyze. No default.")
    subparser_5.add_argument("--model-file", dest="model_file", metavar="SENZING_MODEL_FILE", help="Filename of the trained model. Default: model.pickle")

    return parser

# -----------------------------------------------------------------------------
# Message handling
# -----------------------------------------------------------------------------

# 1xx Informational (i.e. logging.info())
# 2xx Warning (i.e. logging.warn())
# 4xx User configuration issues (either logging.warn() or logging.err() for Client errors)
# 5xx Internal error (i.e. logging.error for Server errors)
# 9xx Debugging (i.e. logging.debug())


message_dictionary = {
    "100": "senzing-" + SENZING_PRODUCT_ID + "{0:04d}I",
    "101": "Enter {0}",
    "102": "Exit {0}",
    "103": "Phrase: '{0}' Category: '{1}'",
    "104": "Percent: {0:.2f}% Category: '{1}'",
    "199": "{0}",
    "200": "senzing-" + SENZING_PRODUCT_ID + "{0:04d}W",
    "400": "senzing-" + SENZING_PRODUCT_ID + "{0:04d}E",
    "498": "Bad SENZING_SUBCOMMAND: {0}.",
    "499": "No processing done.",
    "500": "senzing-" + SENZING_PRODUCT_ID + "{0:04d}E",
    "501": "Error: {0} for {1}",
    "599": "Program terminated with error.",
    "900": "senzing-" + SENZING_PRODUCT_ID + "{0:04d}D",
    "999": "{0}",
}


def message(index, *args):
    index_string = str(index)
    template = message_dictionary.get(index_string, "No message for index {0}.".format(index_string))
    return template.format(*args)


def message_generic(generic_index, index, *args):
    index_string = str(index)
    return "{0} {1}".format(message(generic_index, index), message(index, *args))


def message_info(index, *args):
    return message_generic(100, index, *args)


def message_warn(index, *args):
    return message_generic(200, index, *args)


def message_error(index, *args):
    return message_generic(500, index, *args)


def message_debug(index, *args):
    return message_generic(900, index, *args)


def get_exception():
    ''' Get details about an exception. '''
    exception_type, exception_object, traceback = sys.exc_info()
    frame = traceback.tb_frame
    line_number = traceback.tb_lineno
    filename = frame.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, line_number, frame.f_globals)
    return {
        "filename": filename,
        "line_number": line_number,
        "line": line.strip(),
        "exception": exception_object,
        "type": exception_type,
        "traceback": traceback,
    }

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def get_ini_filename(args_dictionary):
    ''' Find the slice-algorithm.ini file in the filesystem.'''

    # Possible locations for slice-algorithm.ini

    filenames = [
        "{0}/slice-algorithm.ini".format(os.getcwd()),
        "{0}/slice-algorithm.ini".format(os.path.dirname(os.path.realpath(__file__))),
        "{0}/slice-algorithm.ini".format(os.path.dirname(os.path.abspath(sys.argv[0]))),
        "/etc/slice-algorithm.ini",
        "/opt/senzing/g2/python/slice-algorithm.ini",
    ]

    # Return first slice-algorithm.ini found.

    for filename in filenames:
        final_filename = os.path.abspath(filename)
        if os.path.isfile(final_filename):
            return final_filename

    # If file not found, return None.

    return None


def get_configuration(args):
    ''' Order of precedence: CLI, OS environment variables, INI file, default.'''
    result = {}

    # Copy default values into configuration dictionary.

    for key, value in configuration_locator.items():
        result[key] = value.get('default', None)

    # "Prime the pump" with command line args. This will be done again as the last step.

    for key, value in args.__dict__.items():
        new_key = key.format(subcommand.replace('-', '_'))
        if value:
            result[new_key] = value

    # Copy INI values into configuration dictionary.

    ini_filename = get_ini_filename(result)
    if ini_filename:

        result['ini_filename'] = ini_filename

        config_parser = configparser.RawConfigParser()
        config_parser.read(ini_filename)

        for key, value in configuration_locator.items():
            keyword_args = value.get('ini', None)
            if keyword_args:
                try:
                    result[key] = config_parser.get(**keyword_args)
                except:
                    pass

    # Copy OS environment variables into configuration dictionary.

    for key, value in configuration_locator.items():
        os_env_var = value.get('env', None)
        if os_env_var:
            os_env_value = os.getenv(os_env_var, None)
            if os_env_value:
                result[key] = os_env_value

    # Copy 'args' into configuration dictionary.

    for key, value in args.__dict__.items():
        new_key = key.format(subcommand.replace('-', '_'))
        if value:
            result[new_key] = value

    # Special case: subcommand from command-line

    if args.subcommand:
        result['subcommand'] = args.subcommand

    # Special case: Change boolean strings to booleans.

    booleans = ['debug', 'pretty']
    for boolean in booleans:
        boolean_value = result.get(boolean)
        if isinstance(boolean_value, str):
            boolean_value_lower_case = boolean_value.lower()
            if boolean_value_lower_case in ['true', '1', 't', 'y', 'yes']:
                result[boolean] = True
            else:
                result[boolean] = False

    # Special case: Change integer strings to integers.

    integers = []
    for integer in integers:
        integer_string = result.get(integer)
        result[integer] = int(integer_string)

    return result


def validate_configuration(config):
    '''Check aggregate configuration from commandline options, environment variables, config files, and defaults.'''

    user_warning_messages = []
    user_error_messages = []

    # Log warning messages.

    for user_warning_message in user_warning_messages:
        logging.warn(user_warning_message)

    # Log error messages.

    for user_error_message in user_error_messages:
        logging.error(user_error_message)

    # Log where to go for help.

    if len(user_warning_messages) > 0 or len(user_error_messages) > 0:
        logging.info(message_info(198))

    # If there are error messages, exit.

    if len(user_error_messages) > 0:
        exit_error(499)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def entry_template(config):
    '''Format of entry message.'''
    config['start_time'] = time.time()

    # FIXME: Redact sensitive info:  Example: database password.

    config_json = json.dumps(config, sort_keys=True)
    return message_info(101, config_json)


def exit_template(config):
    '''Format of exit message.'''
    stop_time = time.time()
    config['stop_time'] = stop_time
    config['elapsed_time'] = stop_time - config.get('start_time', stop_time)

    # FIXME: Redact sensitive info:  Example: database password.

    config_json = json.dumps(config, sort_keys=True)
    return message_info(102, config_json)


def exit_error(index, *args):
    '''Log error message and exit program.'''
    logging.error(message_error(index, *args))
    logging.error(message_error(599))
    sys.exit(1)


def exit_silently():
    '''Exit program.'''
    sys.exit(1)


def common_prolog(config):
    validate_configuration(config)
    logging.info(entry_template(config))

# -----------------------------------------------------------------------------
# Utility functions for algorithm
# -----------------------------------------------------------------------------


def get_generator_from_jsonlines(jsonlines_filename, max_lines=0):
    '''Tricky code.  Uses currying technique. Create a generator function
       that reads a JSONLines file (http://jsonlines.org/) and yields a dictionary.
    '''

    def result_function():
        with open(jsonlines_filename) as jsonlines_file:
            counter = 0
            for line in jsonlines_file:
                counter += 1
                if max_lines and counter <= max_lines:
                    yield json.loads(line)
                else:
                    return

    return result_function


def pretty_print(percent, suggestion):
    if percent > 1.0:
        if percent > 10:
            print("{0:.1f}% - {1}".format(percent, suggestion))
        else:
            print(" {0:.1f}% - {1}".format(percent, suggestion))


def regex_based_predictor(regex_based_predictors, sample):
        for category_id, category_regex in regex_based_predictors.items():
            if bool(re.match(category_regex, sample)):
                return category_id
        return -1

# -----------------------------------------------------------------------------
# do_* functions
#   Common function signature: do_XXX(args)
# -----------------------------------------------------------------------------


def do_prepare(args):
    '''Read from URL-addressable file.'''

    # Get context from CLI, environment variables, and ini files.

    config = get_configuration(args)

    # Perform common initialization tasks.

    common_prolog(config)

    # Get values from configuration.

    jsonlines_file = config.get('jsonlines_file')
    output_directory = config.get('output_directory')

    # Create generators.

    dictionaries = get_generator_from_jsonlines(jsonlines_file, 500000)

    # Create in-memory structure.

    key_values = {}
    for dictionary in dictionaries():
        for key, value in dictionary.items():
            if key not in key_values.keys():
                key_values[key] = []
            key_values[key].append(value)

    # Write output files

    for key, values in key_values.items():
        directory = "{0}/{1}".format(output_directory, key.lower())
        filename = "{0}/{1}.txt".format(directory, key)
        os.makedirs(directory)
        with open(filename, "w") as output_file:
            for value in values:
                output_file.write("{0}\n".format(value))

    # Epilog.

    logging.info(exit_template(config))


def do_train(args):
    '''Read from URL-addressable file.'''

    # Get context from CLI, environment variables, and ini files.

    config = get_configuration(args)

    # Perform common initialization tasks.

    common_prolog(config)

    # Get values from configuration.

    input_directory = config.get('input_directory')
    model_file = config.get('model_file')

    # Load files.

    model = load_files(input_directory)

    # Write the model file.

    pickle.dump(model, open(model_file, "wb"))

    # Epilog.

    logging.info(exit_template(config))


def do_test_phrase(args):
    '''Read from URL-addressable file.'''

    # Get context from CLI, environment variables, and ini files.

    config = get_configuration(args)

    # Perform common initialization tasks.

    common_prolog(config)

    # Get values from configuration.

    test_phrase = config.get('test_phrase')
    model_file = config.get('model_file')

    # Load files.

    training_set = pickle.load(open(model_file, "rb"))

    count_vect = CountVectorizer()
    training_counts = count_vect.fit_transform(training_set.data)
    tfidf_transformer = TfidfTransformer()
    training_tfidf = tfidf_transformer.fit_transform(training_counts)

    # XXX

    clf = MultinomialNB().fit(training_tfidf, training_set.target)

    # Example samples.

    samples = [test_phrase]

    # Calculate feature_based_predictions of samples.

    sample_counts = count_vect.transform(samples)
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    feature_based_predictions = clf.predict(sample_tfidf)

    # Print samples and feature_based_predictions.

    for sample, prediction in zip(samples, feature_based_predictions):
        logging.info(message_info(103, sample, training_set.target_names[prediction]))

    # Epilog.

    logging.info(exit_template(config))


def do_suggest(args):
    '''Read from URL-addressable file.'''

    # Get context from CLI, environment variables, and ini files.

    config = get_configuration(args)

    # Perform common initialization tasks.

    common_prolog(config)

    # Get values from configuration.

    input_file = config.get('input_file')
    model_file = config.get('model_file')
    pretty = config.get('pretty')

    # Load files.

    training_set = pickle.load(open(model_file, "rb"))

    # Initialize machine learning.

    count_vect = CountVectorizer()
    training_counts = count_vect.fit_transform(training_set.data)
    tfidf_transformer = TfidfTransformer()
    training_tfidf = tfidf_transformer.fit_transform(training_counts)
    clf = MultinomialNB().fit(training_tfidf, training_set.target)

    # Example samples.

    samples = open(input_file).read().splitlines()

    # Calculate regex_based_predictions of samples.

    category_to_id_dictionary = { training_set.target_names[i] : i for i in range(0, len(training_set.target_names)) }
    regex_based_predictors = {
        category_to_id_dictionary.get('cc_account_number'): r'^((4\d{3})|(5[1-5]\d{2})|(6011))-?\d{4}-?\d{4}-?\d{4}|3[4,7]\d{13}$',
        category_to_id_dictionary.get('gender'): r'^(?:m|M|male|Male|f|F|female|Female)$',
        category_to_id_dictionary.get('ssn_number'): r'^\d{3}-\d{2}-\d{4}$',
    }

    regex_based_predictions = []
    for sample in samples:
        regex_based_predictions.append(regex_based_predictor(regex_based_predictors, sample))

    # Calculate feature_based_predictions of samples.

    sample_counts = count_vect.transform(samples)
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    feature_based_predictions = clf.predict(sample_tfidf)

    # Tally predictions.

    min_size = min(len(samples), len(feature_based_predictions))
    prediction_counters = { i : 0 for i in range(0, len(training_set.target_names)) }

    for regex_based_prediction, feature_based_prediction in zip(regex_based_predictions, feature_based_predictions):
        if regex_based_prediction > 0:
            prediction = regex_based_prediction
        else:
            prediction = feature_based_prediction
        prediction_counters[prediction] += 1

    # Calculate percentages.

    prediction_percentages = {}
    for prediction, prediction_count in prediction_counters.items():
        if prediction_count > 0:
            prediction_percentages[training_set.target_names[prediction]] = float(prediction_count) / min_size

    # Print predictions.

    for key, value in sorted(prediction_percentages.items(), reverse=True, key=lambda item: item[1]):
        percent = value * 100
        if pretty:
            pretty_print(percent, key)
        else:
            logging.info(message_info(104, percent, key))

    # Epilog.

    logging.info(exit_template(config))


def do_suggest_as_markdown(args):
    '''Read from URL-addressable file.'''

    # Get context from CLI, environment variables, and ini files.

    config = get_configuration(args)

    # Perform common initialization tasks.

    validate_configuration(config)

    # Get values from configuration.

    input_file = config.get('input_file')
    model_file = config.get('model_file')
    pretty = config.get('pretty')

    # Load files.

    training_set = pickle.load(open(model_file, "rb"))

    count_vect = CountVectorizer()
    training_counts = count_vect.fit_transform(training_set.data)
    tfidf_transformer = TfidfTransformer()
    training_tfidf = tfidf_transformer.fit_transform(training_counts)

    # XXX

    clf = MultinomialNB().fit(training_tfidf, training_set.target)

    # Example samples.

    samples = open(input_file).read().splitlines()

    # Calculate feature_based_predictions of samples.

    sample_counts = count_vect.transform(samples)
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    feature_based_predictions = clf.predict(sample_tfidf)

    print(feature_based_predictions)

    # Print samples and feature_based_predictions.

#     for sample, prediction in zip(samples, feature_based_predictions):
#         logging.info(message_info(103, sample, training_set.target_names[prediction]))

    min_size = min(len(samples), len(feature_based_predictions))
    prediction_counters = {}
    for sample, prediction in zip(samples, feature_based_predictions):
        if prediction not in prediction_counters.keys():
            prediction_counters[prediction] = 0
        prediction_counters[prediction] += 1

    print(prediction_counters)

    prediction_percentages = {}
    for prediction, prediction_count in prediction_counters.items():
        prediction_percentages[training_set.target_names[prediction]] = float(prediction_count) / min_size

    print(prediction_percentages)

    print("\n\n### {0}".format(input_file))
    print("\n```console")
    for key, value in sorted(prediction_percentages.items(), reverse=True, key=lambda item: item[1]):
        percent = value * 100
        pretty_print(percent, key)
    print("```")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    # Configure logging. See https://docs.python.org/2/library/logging.html#levels

    log_level_map = {
        "notset": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "fatal": logging.FATAL,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    log_level_parameter = os.getenv("SENZING_LOG_LEVEL", "info").lower()
    log_level = log_level_map.get(log_level_parameter, logging.INFO)
    logging.basicConfig(format=log_format, level=log_level)

    # Parse the command line arguments.

    subcommand = os.getenv("SENZING_SUBCOMMAND", None)
    parser = get_parser()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        subcommand = args.subcommand
    elif subcommand:
        args = argparse.Namespace(subcommand=subcommand)
    else:
        parser.print_help()
        exit_silently()

    # Transform subcommand from CLI parameter to function name string.

    subcommand_function_name = "do_{0}".format(subcommand.replace('-', '_'))

    # Test to see if function exists in the code.

    if subcommand_function_name not in globals():
        logging.warn(message_warn(498, subcommand))
        parser.print_help()
        exit_silently()

    # Tricky code for calling function based on string.

    globals()[subcommand_function_name](args)
