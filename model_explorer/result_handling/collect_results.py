import os
import glob
from model_explorer.result_handling.results_collection import ResultsCollection
from model_explorer.utils.logger import logger


def collect_results(path: str) -> ResultsCollection:
    """This function automatically gathers all results found in a given path.

    Args:
        path (str): Pathlike object, which is searched for results, can be a
        single file or a folder with multiple pickles

    Returns:
        ResultsCollection: all found results
    """
    if os.path.isdir(path):
        # if path start with empty results loader
        results_collection = ResultsCollection()
        for result_file in glob.glob(os.path.join(path, '*.pkl')):
            rl = ResultsCollection(pickle_file=result_file)
            if results_collection.individuals == []:
                results_collection = rl
            else:
                results_collection.merge(rl)
            logger.debug("Added results file: {} with {} individual(s)".format(result_file, len(rl.individuals)))
    else:
        results_collection = ResultsCollection(path)
        logger.debug("Added results file: {} with {} individual(s)".format(path, len(results_collection.individuals)))

    results_collection.drop_duplicate_parameters()
    logger.debug("Loaded in total {} individuals".format(len(results_collection.individuals)))

    return results_collection
