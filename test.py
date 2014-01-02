import models, utils
import logging, sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=='__main__':
    if len(sys.argv) == 1:
        print "No model name was given! Run again using format: \n\t",
        print "python test.py modelname"
    else:
        modelname = sys.argv[1]
        pred = models.test_model(modelname)
        name = modelname + ".csv"
        utils.create_submission(name, pred)
        print "Saved submission with name %s" %(name)
