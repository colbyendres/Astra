import threading 

from flask import Flask
from RecommendationEngine import RecommendationEngine, PaperIndex, Papers
from config import Config
from models import db
from routes import bp as routes_bp
from background_tasks import cache_faiss_index
    
def start_app():
    # Pull in FAISS index from S3 as a background job
    threading.Thread(target=cache_faiss_index, daemon=True).start()
    
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DB_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.secret_key = Config.SESSION_KEY
    db.init_app(app)
    
    index = PaperIndex(Config.LOCAL_FAISS_PATH)
    papers = Papers(db.session)
    rec = RecommendationEngine(index, papers)
    with app.app_context():
        app.recommender = rec
        app.papers = papers
    app.register_blueprint(routes_bp)
    return app

flask_app = start_app()
