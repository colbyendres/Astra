from flask import Blueprint, render_template, request , current_app, flash, get_flashed_messages
from requests import HTTPError

bp = Blueprint('main', __name__)

@bp.route("/", methods=["GET"])
def home():
    paper_estimate = current_app.recommender.get_total_papers()
    return render_template("home.html", num_papers = paper_estimate)

@bp.route("/search", methods=["GET"])
def search():
    return render_template("search.html")

@bp.route("/results", methods=["POST"])
def results():
    query_title = request.form["title"]
    k = int(request.form["k"])
    try:
        recommendations = current_app.recommender.recommend(query_title, k)
        return render_template("results.html", query=query_title, recommendations=recommendations)
    except HTTPError as e:
        flash(f'Unable to parse title (HTTP code {e.response.status_code})', 'error')
        return render_template("search.html")