from flask import Flask
from RecommendationEngine import RecommendationEngine
from config import Config
from models import db
from routes import bp as routes_bp

import boto3 

def start_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DB_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.secret_key = Config.SESSION_KEY
    db.init_app(app)

    session = boto3.Session(
        aws_access_key_id=Config.BUCKETEER_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.BUCKETEER_AWS_SECRET_ACCESS_KEY,
        region_name=Config.BUCKETEER_AWS_REGION
    )
    client = session.client('s3')
    with app.app_context():
        app.recommender = RecommendationEngine(
            db_session=db.session,
            s3_client=client
        )

    app.register_blueprint(routes_bp)
    return app


app = start_app()
