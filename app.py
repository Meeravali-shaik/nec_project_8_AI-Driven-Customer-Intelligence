from flask import Flask

app = Flask(__name__)
app.secret_key = "customer-intelligence-dev"

from routes.dataset import *
from routes.dashboard import *
from routes.customers import *
from routes.segmentation import *
from routes.prediction import *
from routes.churn import *
from routes.classification import *

from routes.recommendations import *
from routes.behavior_analysis import *
from routes.market_analysis import *
from routes.emi_analysis import *
from routes.tenure_analysis import *
from routes.expiry_monitor import *
from routes.reports import *

if __name__ == "__main__":
    app.run(debug=True)
