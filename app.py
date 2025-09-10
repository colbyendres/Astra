from flask import Flask
from RecommendationEngine import RecommendationEngine
from config import Config
from models import db
from routes import bp as routes_bp

def start_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DB_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)

    with app.app_context():
        app.recommender = RecommendationEngine(
            faiss_abstract_idx=Config.EMBEDDINGS_INDEX,
            db_session=db.session
        )

    app.register_blueprint(routes_bp)
    return app


app = start_app()
